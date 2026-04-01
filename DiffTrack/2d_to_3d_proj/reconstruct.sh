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

set -euo pipefail

#run metric depth (unidepth)
for seq in ${evalset[@]}; do
    SEQ_DIR=$PROJECT_ROOT/test_data/$seq
    FRAME_DIR=$SEQ_DIR/frames
    Depth_DIR=$SEQ_DIR/processed_data
    [ -d "$FRAME_DIR" ] || { echo "Missing frames dir: $FRAME_DIR"; exit 1; }

    CUDA_VISIBLE_DEVICES=3 python UniDepth/scripts/demo_mega-sam.py \
    --scene-name $seq \
    --img-path $FRAME_DIR \
    --outdir $Depth_DIR/UniDepth/ #for output saving
done



# Run DepthAnything (relative)
for seq in ${evalset[@]}; do
    SEQ_DIR=$PROJECT_ROOT/test_data/$seq
    FRAME_DIR=$SEQ_DIR/frames
    Depth_DIR=$SEQ_DIR/processed_data
    [ -d "$FRAME_DIR" ] || { echo "Missing frames dir: $FRAME_DIR"; exit 1; }

    CUDA_VISIBLE_DEVICES=3 python Depth-Anything/run_videos.py --encoder vitl \
    --load-from $Anything_weight \
    --img-path $FRAME_DIR \
    --outdir $Depth_DIR/Depth-Anything/
done



#run DROID SLAM (?)
for seq in ${evalset[@]}; do
    SEQ_DIR=$PROJECT_ROOT/test_data/$seq
    FRAME_DIR=$SEQ_DIR/frames
    Depth_DIR=$SEQ_DIR/processed_data
    [ -d "$FRAME_DIR" ] || { echo "Missing frames dir: $FRAME_DIR"; exit 1; }

    MONO_BASE=$Depth_DIR/Depth-Anything
    METRIC_BASE=$Depth_DIR/UniDepth

    MONO_DIR=$MONO_BASE
    METRIC_DIR=$METRIC_BASE

    # Handle both layouts:
    # 1) .../Depth-Anything/<files>
    # 2) .../Depth-Anything/$seq/<files>
    if [ -d "$MONO_BASE/$seq" ]; then
        MONO_DIR=$MONO_BASE/$seq
    fi
    if [ -d "$METRIC_BASE/$seq" ]; then
        METRIC_DIR=$METRIC_BASE/$seq
    fi

    # Fail early if depth outputs are empty/missing
    MONO_COUNT=$(find "$MONO_DIR" -maxdepth 1 -type f | wc -l || true)
    METRIC_COUNT=$(find "$METRIC_DIR" -maxdepth 1 -type f | wc -l || true)
    [ "$MONO_COUNT" -gt 0 ] || { echo "No mono depth files in: $MONO_DIR"; exit 1; }
    [ "$METRIC_COUNT" -gt 0 ] || { echo "No metric depth files in: $METRIC_DIR"; exit 1; }

    CUDA_VISIBLE_DEVICES=3 python3 camera_tracking_scripts/test_demo.py \
    --datapath=$FRAME_DIR \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path $MONO_DIR \
    --metric_depth_path $METRIC_DIR \
    --disable_vis
done


# Run Raft Optical Flows
for seq in ${evalset[@]}; do
    SEQ_DIR=$PROJECT_ROOT/test_data/$seq
    FRAME_DIR=$SEQ_DIR/frames
    [ -d "$FRAME_DIR" ] || { echo "Missing frames dir: $FRAME_DIR"; exit 1; }

    CUDA_VISIBLE_DEVICES=3 python3 cvd_opt/preprocess_flow.py \
    --datapath=$FRAME_DIR \
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