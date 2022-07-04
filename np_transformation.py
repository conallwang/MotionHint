import numpy as np
from scipy.spatial.transform import Rotation as R

def kittiformat2SEs(kitti_data):
    N = kitti_data.shape[0]

    SEs = np.zeros((N, 4, 4))
    for i in range(N):
        SE = np.eye(4)
        SE[:3, :] = kitti_data[i].reshape(3, 4)
        SEs[i] = SE
    return SEs

def SEs2kittiformat(SE_datas):
    N = SE_datas.shape[0]

    kitti_datas = np.zeros((N, 12))
    for i in range(N):
        kitti_datas[i] = SE_datas[i][:3, :].reshape(12)
    return kitti_datas

def SO2so(SO_data):
    return R.from_matrix(SO_data).as_rotvec()

def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix()

def SE2se(SE_data):
    se = np.zeros(6)
    se[:3] = SE_data[:3, 3]
    se[3:] = SO2so(SE_data[:3, :3])
    return se

def SEs2ses(SEs_data):
    N = SEs_data.shape[0]

    ses = np.zeros((N, 6))
    for i in range(N):
        ses[i] = SE2se(SEs_data[i])
    return ses

def se2SE(se_data):
    SE = np.eye(4)
    SE[:3, :3] = so2SO(se_data[3:])
    SE[:3, 3] = se_data[:3]
    return SE

def ses2SEs(ses_data):
    N = ses_data.shape[0]

    SEs = np.zeros((N, 4, 4))
    for i in range(N):
        SEs[i] = se2SE(ses_data[i])
    return SEs
