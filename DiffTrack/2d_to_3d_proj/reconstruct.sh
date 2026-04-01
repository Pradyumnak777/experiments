#FROM Instant4D paper!!

#make sure to run from '2d_to_3d_proj' directory

# directories to reconstruct
evalset=(
  benchpress
)

# replace with own paths
PROJECT_ROOT=$(pwd)
CKPT_PATH=$PROJECT_ROOT/mega-sam/checkpoints/megasam_final.pth
Anything_weight=$PROJECT_ROOT/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth
raft_ckpt=$PROJECT_ROOT/mega-sam/cvd_opt/raft-things.pth

cd mega-sam  

export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

#run metric depth (unidepth)
for seq in ${evalset[@]}; do
    DATA_DIR=$PROJECT_ROOT/$seq/test_data
    Depth_DIR=$PROJECT_ROOT/$seq/test_data/processed_data
    CUDA_VISIBLE_DEVICES=3 python UniDepth/scripts/demo_mega-sam.py \
    --scene-name $seq \
    --img-path $DATA_DIR \
    --outdir $Depth_DIR/UniDepth/ #for output saving
done



# Run DepthAnything (relative)
for seq in ${evalset[@]}; do
    DATA_DIR=$PROJECT_ROOT/$seq/test_data
    Depth_DIR=$PROJECT_ROOT/$seq/test_data/processed_data
    CUDA_VISIBLE_DEVICES=3 python Depth-Anything/run_videos.py --encoder vitl \
    --load-from $Anything_weight \
    --img-path $DATA_DIR \
    --outdir $Depth_DIR/Depth-Anything/$seq
done



#run DROID SLAM (?)
for seq in ${evalset[@]}; do
    DATA_DIR=$PROJECT_ROOT/$seq/test_data
    Depth_DIR=$PROJECT_ROOT/$seq/test_data/processed_data
    CUDA_VISIBLE_DEVICES=3 python3 camera_tracking_scripts/test_demo.py \
    --datapath=$DATA_DIR \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path  $Depth_DIR/Depth-Anything \
    --metric_depth_path $Depth_DIR/UniDepth \
    --disable_vis
done


# Run Raft Optical Flows
for seq in ${evalset[@]}; do
    DATA_DIR=$PROJECT_ROOT/$seq/test_data
    Depth_DIR=$PROJECT_ROOT/$seq/test_data/processed_data
    CUDA_VISIBLE_DEVICES=3 python3 cvd_opt/preprocess_flow.py \
    --datapath=$DATA_DIR \
    --model=$raft_ckpt \
    --scene_name $seq --mixed_precision
done

# Run CVD optimization (#combines all info to form some sort of uniform depth measure(?))
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=0 python3 cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0
done

cd ../..