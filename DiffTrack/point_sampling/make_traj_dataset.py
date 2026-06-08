'''
build a trajectory dataset over the whole training set
1. for each video: flowfeat on first frame -> sample semantic points
2. track those points through the (frame-capped) video with co-tracker
3. save the trajectories to disk, to feed the network later
'''

import os
import glob
import pickle
import torch
import numpy as np
import cv2
from flowfeat_mask import load_flowfeat, get_flowfeat, load_video_frames
from cluster_sample import sample_semantic_points

TRAIN_DIR  = 'UCF_Rep/train'
OUT_DIR    = 'point_sampling/traj_dataset'
M          = 256          #points to sample per video
CLUSTERS   = 16
SIZE       = (224, 224)   #flowfeat + tracking resolution (keep matched so coords line up)
MAX_FRAMES = 64           #cap frames fed to co-tracker (bounds gpu memory + uniform T)


def load_full_video(mp4_path):
    #load frames at 224x224 as [1, T, 3, H, W] in [0,255], uniformly capped to MAX_FRAMES
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, SIZE)
        frames.append(frame)
    cap.release()

    #uniformly subsample if the clip is longer than MAX_FRAMES (frame 0 always kept)
    if len(frames) > MAX_FRAMES:
        idx    = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
        frames = [frames[i] for i in idx]

    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    return video.unsqueeze(0)   #[1, T, 3, H, W]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading FlowFeat...")
    model = load_flowfeat(device=device)

    print("Loading CoTracker...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

    video_paths = sorted(glob.glob(os.path.join(TRAIN_DIR, '*.mp4')))
    print(f"found {len(video_paths)} videos")

    for i, path in enumerate(video_paths):
        name     = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUT_DIR, f'{name}.pkl')
        if os.path.exists(out_path):
            print(f"[{i+1}/{len(video_paths)}] skip {name} (already done)")   #resume-friendly
            continue

        print(f"[{i+1}/{len(video_paths)}] {name}")

        try:
            #flowfeat on first frame -> sample points
            pixels = load_video_frames(path).to(device)     #[1, 1, 3, 224, 224]
            feats  = get_flowfeat(pixels, model)            #[1, 1, 128, 224, 224]
            Ps     = sample_semantic_points(feats, M=M, clusters=CLUSTERS)   #[M, 2] (x, y)

            #track the sampled points through the (capped) video
            video   = load_full_video(path).to(device)      #[1, T, 3, 224, 224]
            t       = torch.zeros(Ps.shape[0], 1, device=device)   #all queried at frame 0
            queries = torch.cat([t, Ps], dim=1).unsqueeze(0)       #[1, M, 3] = (t, x, y)

            with torch.no_grad():
                tracks, visibility = cotracker(video, queries=queries)   #[1, T, M, 2], [1, T, M]

            #save the trajectories
            with open(out_path, 'wb') as f:
                pickle.dump({
                    'video_path': path,
                    'tracks':     tracks.detach().cpu().numpy(),       #[1, T, M, 2]
                    'visibility': visibility.detach().cpu().numpy(),   #[1, T, M]
                }, f)

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on {name}, skipping")   #don't let one video kill the run

        finally:
            #free everything between videos so memory doesn't accumulate
            for v in ['pixels', 'feats', 'Ps', 'video', 'queries', 'tracks', 'visibility']:
                if v in dir():
                    del v
            torch.cuda.empty_cache()

    print("done.")