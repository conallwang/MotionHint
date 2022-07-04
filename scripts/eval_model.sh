#!/bin/bash
cd $(dirname $0)
cd ..

DATA_PATH=~/ssd/color

MODEL_DIR=$1
# OUT_DIR=./evaluations/result/$(date "+%Y%m%d_%H%M")
OUT_DIR=./evaluations/result/$2     # set output name manually

mkdir $OUT_DIR

# generate predicted pose file
for ((i=9; i<=10; i++))
do
    seq=$i
    if [ $i -lt 10 ]; then
        seq=0$i
    fi
    echo computing $seq ...

    python evaluate_pose.py --eval_split odom_$i --load_weights_folder $MODEL_DIR --data_path $DATA_PATH
    cp $MODEL_DIR/odom_$seq/pred_poses.txt $OUT_DIR/$seq.txt
done

# evaluation
cd ./evaluations
python eval_odom.py --result ./result/$2 --align 7dof
