# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import shutil
from np_transformation import SEs2ses
from datasets.kitti_dataset import KITTIOdomDataset
from rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from multilossmanager import MultiLossManager
from posemanager import PoseManager
from networks.PPnet import PPnet
from operator import itemgetter
from tqdm import tqdm

import numpy as np
import time
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.best_val = float('inf')
        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        ## MotionHint: PPnet
        self.models["ppnet"] = networks.PPnet().to(self.device)
        if self.opt.ppnet:
            self.models["ppnet"].load_state_dict(torch.load(self.opt.ppnet))
        else:
            print("You must provide the path of PPnet.")
            sys.exit(1)
        # end

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        ## MotionHint: build Pose Manager
        self.pseudo_poses_initialized = False
        self.pseudo_poses = {}
        for lines in train_filenames:
            seq, frame_id, side = lines.split()
            frame_id = int(frame_id)
            if (seq, side) not in self.pseudo_poses:
                self.pseudo_poses[(seq, side)] = frame_id
            else:
                self.pseudo_poses[(seq, side)] = max(self.pseudo_poses[(seq, side)], frame_id)
        
        for key in self.pseudo_poses:
            max_num = self.pseudo_poses[key] + 2
            self.pseudo_poses[key] = PoseManager(torch.zeros(max_num, 2, 6).to(self.device))
            self.pseudo_poses[key][:, 1, 0] = float('inf')
        # end

        if self.opt.load_weights_folder is not None:
            self.load_model()

        ## MotionHint: convient to set learning rate
        for param_group in self.model_optimizer.param_groups:
            param_group["lr"] = self.opt.learning_rate
        # end

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        print(f"Current Learning Rate: {self.model_optimizer.param_groups[0]['lr']}")

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        ## MotionHint: Using 'Multi Loss Rebalancing' for weights initialization, just update once
        self.num_losses = 2
        self.mlm = MultiLossManager(self.opt.batch_size, self.num_losses, self.opt.num_for_rebalance, self.opt.update_once)
        # end

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    ## MotionHint
    def initialize(self):
        """Generate prior poses using the original SSM-VO
        """
        print('=> Initialize the pseudo poses')
        self.set_eval()

        initialize_bar = tqdm(range(len(self.train_loader)))
        for inputs in self.train_loader:
            _, _ = self.process_batch(inputs)
            initialize_bar.update()
        
        save_path = os.path.join(self.log_path, 'pseudo_poses.pth')
        torch.save(self.pseudo_poses, save_path)
        print(f'=> The pseudo poses have been saved in {save_path}')
    # end

    def train(self):
        """Run the entire training pipeline
        """
        ## MotionHint:
        ## If the Pose Manager has not been initialized, call 'initialize' function
        if not self.pseudo_poses_initialized:
            with torch.no_grad():
                self.initialize()
        # end

        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            ## MotionHint: Param lambda of multi loss rebalancing
            self.current_lambda = max(self.opt.lambda_start + self.epoch * self.opt.lambda_slope, self.opt.lambda_end)
            # end
            self.run_epoch()
            self.val()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        epoch_loss = 0
        origin_loss = 0
        motion_loss = 0
        sample_num = 0
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            ## MotionHint: Combining the origin loss and the motion loss using Multi Loss Rebalancing
            with torch.no_grad():
                motion_index = losses['motion_loss'] > 0
                num_motion = motion_index.sum()
            if num_motion:
                loss, cur_ptr = self.mlm.get_total_loss([losses['origin_loss'][motion_index], losses['motion_loss'][motion_index]], num_motion, True)

                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()

                if cur_ptr >= self.opt.num_for_rebalance:
                    self.mlm.rebalancing(self.current_lambda, self.epoch, os.path.join(self.log_path, 'weights.txt'))
                
                epoch_loss += loss.cpu().data * num_motion
                origin_loss += losses["origin_loss"][motion_index].sum().data
                motion_loss += losses["motion_loss"][motion_index].sum().data
                sample_num += num_motion
            else:
                loss, cur_ptr = self.mlm.get_total_loss([losses['origin_loss'], losses['motion_loss']], self.opt.batch_size, False)

                # not update parameters, just clear calculation graph
                self.model_optimizer.zero_grad()
                loss.backward()
            # end

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, loss.cpu().data)

            self.step += 1

        if sample_num > 0:
            epoch_loss /= sample_num
            origin_loss /= sample_num
            motion_loss /= sample_num

        print(f'Total Loss: {epoch_loss}, Origin Loss: {origin_loss}, Motion Loss: {motion_loss}')
        with open(f'{self.log_path}/log.txt', 'a') as f:
            f.write(f'Total Loss: {epoch_loss}, Origin Loss: {origin_loss}, Motion Loss: {motion_loss} \n')

    def process_batch(self, inputs, is_training=True):
        """Pass a minibatch through the network and generate images and losses
        """
        # skip 'filename' items
        for key, ipt in inputs.items():
            if isinstance(ipt, list):
                continue
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, is_training)

        ## MotionHint: 
        ## Update prior poses in Pose Manager, if new pose have lower loss value.
        with torch.no_grad():
            if is_training:
                self.update_pseudo_poses(inputs, outputs, losses['origin_loss'].data)
        # end

        return outputs, losses

    def update_pseudo_poses(self, inputs, outputs, judge_losses):
        """Update poses in Pose Manager according to loss
        """
        for i in range(self.opt.batch_size):
            seq, frame_id, side = inputs['filename'][i].split()
            frame_id = int(frame_id)

            T_10 = outputs['cam_T_cam', 0, 1][i].data
            T_n10 = outputs['cam_T_cam', 0, -1][i].data
            judge_loss = judge_losses[i]

            loss0 = self.pseudo_poses[(seq, side)][frame_id, 1, 0]
            loss1 = self.pseudo_poses[(seq, side)][frame_id + 1, 1, 0]
            if judge_loss < loss0:
                self.pseudo_poses[(seq, side)].update(frame_id, self.SE2se(T_n10.inverse()), judge_loss)
            
            if judge_loss < loss1:
                self.pseudo_poses[(seq, side)].update(frame_id+1, self.SE2se(T_10), judge_loss)

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    # def val(self):
    #     """Validate the model on a single minibatch
    #     """
    #     self.set_eval()
        
    #     eval_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #     img_ext = '.png' if self.opt.png else '.jpg'

    #     ates = []
    #     for seq_id in eval_list:
    #         print(f'=> Eval seq {seq_id} ...')
    #         filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", "odom",
    #                  "test_files_{:02d}.txt".format(seq_id)))
    #         dataset = KITTIOdomDataset(self.opt.data_path, filenames, self.opt.height, self.opt.width,
    #                                 [0, 1], 4, is_train=False, img_ext=img_ext)
    #         dataloader = DataLoader(dataset, self.opt.batch_size, shuffle=False,
    #                     num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
            
    #         bar = tqdm(range(len(dataloader)))
    #         pred_local_mat = []
    #         with torch.no_grad():
    #             for inputs in dataloader:
    #                 for key, ipt in inputs.items():
    #                     if isinstance(ipt, list):
    #                         continue
    #                     inputs[key] = ipt.cuda()

    #                 all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in [0, 1]], 1)

    #                 features = [self.models['pose_encoder'](all_color_aug)]
    #                 axisangle, translation = self.models['pose'](features)

    #                 pred_local_mat.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
    #                 bar.update()
            
    #         pred_local_mat = np.concatenate(pred_local_mat)
    #         pred_global_mat = relative2absolute(pred_local_mat)

    #         gt_poses_path = os.path.join(self.opt.data_path, "poses", "{:02d}.txt".format(seq_id))
    #         gt_global_mat = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    #         gt_global_mat = np.concatenate(
    #             (gt_global_mat, np.zeros((gt_global_mat.shape[0], 1, 4))), 1)
    #         gt_global_mat[:, 3, 3] = 1

    #         pred_global_poses = SEs2ses(pred_global_mat)
    #         gt_global_poses = SEs2ses(gt_global_mat)

    #         r, t, scale = umeyama_alignment(pred_global_poses[:, :3].transpose(1, 0), gt_global_poses[:, :3].transpose(1, 0), True)
            
    #         align_transformation = np.eye(4)
    #         align_transformation[:3:, :3] = r
    #         align_transformation[:3, 3] = t

    #         N = pred_global_mat.shape[0]
    #         for i in range(N):
    #             pred_global_mat[i][:3, 3] *= scale
    #             pred_global_mat[i] = align_transformation @ pred_global_mat[i]

    #         errors = []
    #         for i in range(N):
    #             gt_xyz = gt_global_mat[i][:3, 3]
    #             pred_xyz = pred_global_mat[i][:3, 3]

    #             align_err = gt_xyz - pred_xyz

    #             errors.append(np.sqrt(np.sum(align_err ** 2)))
    #         ates.append(np.sqrt(np.mean(np.asarray(errors) ** 2)))

    #     mean_ate = np.mean(ates)
    #     with open(f'{self.log_path}/log.txt', 'a') as f:
    #         f.write('ATE: [')
    #         for ate in ates:
    #             f.write(f' {ate},')
    #         f.write(']\n')
    #         f.write(f'Mean ATE: {mean_ate}\n')

    #     self.set_train()

    def val(self):
        """Validate the model
        """
        self.set_eval()
        val_loss = 0
        vbar = tqdm(range(len(self.val_loader)))
        while True:
            try:
                inputs = self.val_iter.next()
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                break

            with torch.no_grad():
                outputs, losses = self.process_batch(inputs, False)

                val_loss += losses['origin_loss'].mean().cpu().data
            vbar.update()

        val_loss /= len(self.val_loader)
        with open(f'{self.log_path}/log.txt', 'a') as f:
            f.write(f'Validate Loss: {val_loss}\n')

        if (self.epoch + 1) % self.opt.save_frequency == 0:
            is_best = val_loss <= self.best_val
            if is_best:
                self.best_val = val_loss
            self.save_model(is_best)

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def use_weak_supervised(self, inputs):
        """Check if can use motion supervision. (if there is enough consecutive prior poses to predict the pseudo pose)
        """
        judge_mat = torch.zeros((self.opt.batch_size, 20)).to(self.device)
        judge_mat[:] = float('inf')

        for i in range(self.opt.batch_size):
            seq, frame_id, side = inputs['filename'][i].split()
            frame_id = int(frame_id)

            if frame_id < 20:
                continue

            judge_mat[i] = self.pseudo_poses[(seq, side)][frame_id-19:frame_id+1, 1, 0]
        
        use_weak_supervision = (judge_mat == float('inf')).sum(1) == 0
        return use_weak_supervision

    def se2SE(self, se_data):
        SE = torch.eye(4).to(self.device)
        SE[:3, :3] = axis_angle_to_matrix(se_data[3:])
        SE[:3, 3] = se_data[:3]
        return SE

    def ses2SEs(self, se_datas):
        N = se_datas.shape[0]

        SEs = torch.zeros(N, 4, 4).to(self.device)
        for i in range(N):
            SEs[i] = self.se2SE(se_datas[i])
        return SEs

    def SE2se(self, SE_data):
        se = torch.zeros(6).to(self.device)
        se[:3] = SE_data[:3, 3]
        se[3:] = matrix_to_axis_angle(SE_data[:3, :3])
        return se

    def SEs2ses(self, SE_datas):
        N = SE_datas.shape[0]

        ses = torch.zeros(N, 6).to(self.device)
        for i in range(N):
            ses[i] = self.SE2se(SE_datas[i])
        return ses

    # pos_seqs: [BS, 19, 6], relative poses
    def translate_poses(self, pos_seqs, mode='middle'):
        b, num, dim = pos_seqs.shape

        I_idx = (num+1) // 2
        if mode != 'middle':
            print('You need implemente other modes by yourself !')
            return None

        pos_seqs_mat = self.ses2SEs(pos_seqs.reshape(-1, 6)).reshape(b, -1, 4, 4)       # [BS, 19, 4, 4]
        translated_pos_mat = torch.zeros(b, num+1, 4, 4).to(self.device)                # [BS, 20, 4, 4]

        translated_pos_mat[:, I_idx] = torch.eye(4)
        for i in range(I_idx-1, -1, -1):
            translated_pos_mat[:, i] = translated_pos_mat[:, i+1] @ pos_seqs_mat[:, i]
        for i in range(I_idx+1, num+1):
            translated_pos_mat[:, i] = translated_pos_mat[:, i-1] @ pos_seqs_mat[:, i-1].inverse()
        
        translated_poses = self.SEs2ses(translated_pos_mat.reshape(-1, 4, 4)).reshape(b, -1, 6)
        
        return translated_poses

    def get_pseudo_poses(self, keys, frame_ids, use_weak_supervision):
        """Obtain the pseudo label

        1. predict the pseudo pose using PPnet
        2. compute the pseudo label
        """
        poselist = itemgetter(*keys)(self.pseudo_poses)

        # Note that the poses saved in Pose Manager are relative poses
        poses = torch.zeros(self.opt.batch_size, 19, 6).to(self.device)
        for i in range(self.opt.batch_size):
            if use_weak_supervision[i]:
                target_frame_id = frame_ids[i]
                poses[i] = poselist[i][target_frame_id-19:target_frame_id, 0]
        
        # Convert relative poses to pose-centralized poses
        translated_poses = self.translate_poses(poses)      # [BS, 20, 6]

        # 1. predict the pseudo pose using PPnet
        mean, log_variance = self.models['ppnet'](translated_poses)

        # 2. compute the pseudo label
        relative_mat_label = self.ses2SEs(mean).inverse() @ self.ses2SEs(translated_poses[:, -1])
        
        variance = torch.exp(log_variance)

        return relative_mat_label, variance

    def get_confidence(self, uncertainty, coff=1e3):
        """Convert the predicted variance to supervision confidence
        """
        base = torch.tensor(coff).to(self.device)
        return torch.pow(base, -uncertainty*coff)

    def get_weak_supervision(self, inputs, outputs, side_id, use_weak_supervision):
        """Compute the motion supervision

        1. predict the pseudo pose using PPnet
        2. compute the pseudo label of ego-motion
        3. compute the weighted difference as the motion supervision
        """
        lines = [line.split() for line in inputs['filename']]

        keys = [(line[0], line[2]) for line in lines]
        frame_ids = [int(line[1]) for line in lines]

        if side_id > 0:
            frame_ids = [int(line[1]) + 1 for line in lines]
        
        with torch.no_grad():
            # 1. predict the pseudo pose using PPnet
            # 2. compute the pseudo label of ego-motion
            pseudo_mat, uncertainties = self.get_pseudo_poses(keys, frame_ids, use_weak_supervision)

            if side_id < 0:
                pseudo_mat = pseudo_mat.inverse()
        
        pred_mat = outputs['cam_T_cam', 0, side_id]

        pred_poses = self.SEs2ses(pred_mat)
        pseudo_poses = self.SEs2ses(pseudo_mat)

        confidence = self.get_confidence(uncertainties.sum(1))
        confidence_filter = confidence * (confidence > self.opt.conf_thres)

        # 3. compute the weighted difference as the motion supervision
        return confidence_filter * torch.norm(pseudo_poses - pred_poses, dim=1)

    def coff_expand(self, coff, pattern):
        res = torch.zeros_like(pattern).to(self.device)
        for i in range(pattern.shape[0]):
            res[i] = coff[i]
        return res

    def compute_losses(self, inputs, outputs, is_training):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        origin_loss = 0

        ## MotionHint: 
        ## While training
        ## 1. check if can use motion supervision. (At some points, there is not enough prior poses to predicted the pseudo pose)
        ## 2. predict the pseudo pose using PPnet and compute the pseudo label
        if is_training:
            motion_loss = 0
            with torch.no_grad():
                use_weak_supervision = self.use_weak_supervised(inputs)
            if use_weak_supervision.sum():
                weak_supervision = {}
                for frame_id in self.opt.frame_ids[1:]:
                    weak_supervision[frame_id] = self.get_weak_supervision(inputs, outputs, frame_id, use_weak_supervision) * use_weak_supervision
            else:
                weak_supervision = {}
                for frame_id in self.opt.frame_ids[1:]:
                    weak_supervision[frame_id] = torch.zeros(self.opt.batch_size)
        # end

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
                ## MotionHint: build motion supervision
                ## note: 'idxs' determines which side is used to supervise
                if is_training:
                    motion_error = torch.zeros_like(to_optimise).to(self.device)
                    search_n1 = (idxs == 0) | (idxs == 2)
                    search_p1 = (idxs == 1) | (idxs == 3)
                    motion_error[search_n1] = self.coff_expand(weak_supervision[-1], motion_error)[search_n1]
                    motion_error[search_p1] = self.coff_expand(weak_supervision[1], motion_error)[search_p1]
                # end

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean(-1).mean(-1)
            if is_training:
                motion_loss += motion_error.mean(-1).mean(-1)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            origin_loss += loss
            losses["loss/{}".format(scale)] = loss

        origin_loss /= self.num_scales
        losses["origin_loss"] = origin_loss

        if is_training:
            motion_loss /= self.num_scales
            losses["motion_loss"] = motion_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        with open(f'{self.log_path}/log.txt', 'a') as f:
            f.write(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
            f.write('\n')

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, is_best):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        if is_best:
            tgt_folder = os.path.join(self.log_path, "models", "best")
            if not os.path.exists(tgt_folder):
                os.makedirs(tgt_folder)
            
            for file in os.listdir(save_folder):
                src_file = os.path.join(save_folder, file)
                tgt_file = os.path.join(tgt_folder, file)
                if os.path.isfile(src_file):
                    shutil.copyfile(src_file, tgt_file)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
        
        # loading pseudo poses
        pseudo_poses_path = os.path.join(self.opt.load_weights_folder, "pseudo_poses.pth")
        if os.path.isfile(pseudo_poses_path):
            print("Loading Pseudo Poses")
            self.pseudo_poses = torch.load(pseudo_poses_path)
            self.pseudo_poses_initialized = True
        else:
            print("Cannot find Pseudo Poses so execute initializing")
