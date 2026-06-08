'''
1. take input video
2. perform flowfeat to get 128 dim emebedding for every pixel, in every frame
3. perform "clustering"(?), to sample points
'''

import cv2
from flowfeat_mask import load_flowfeat, get_flowfeat  #will return the 128 dim embeddings
from flowfeat_mask import load_video_frames #returns [1, t, 3, h, w]
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import numpy as np

VIDEO_NAME  = 'v_CuttingInKitchen_g21_c01'
VIDEO_PATH  = f'UCF_Rep/val/{VIDEO_NAME}.mp4'

def sample_semantic_points(feats, M, clusters, seed=0):
    np.random.seed(seed)
    _, T, C, H, W = feats.shape
    q = M // clusters   # points per cluster

    # take first frame, reshape to [H*W, 128] for clustering
    frame = feats[0, 0].detach().float()                     # [128, H, W]
    flat  = frame.permute(1, 2, 0).reshape(-1, C)            # [H*W, 128]
    flat  = F.normalize(flat, dim=-1).cpu().numpy()

    # cluster into L motion-semantic groups
    labels = MiniBatchKMeans(n_clusters=clusters, random_state=seed).fit_predict(flat)  # [H*W]

    # sample q points from each cluster
    sampled = []
    for cid in range(clusters):
        idx    = np.where(labels == cid)[0]
        chosen = np.random.choice(idx, size=min(q, len(idx)), replace=False)
        rows, cols = chosen // W, chosen % W
        sampled.extend(zip(cols.tolist(), rows.tolist()))    # (x, y) = (col, row)

    return torch.tensor(sampled[:M], dtype=torch.float32, device=feats.device)  # [M, 2]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading FlowFeat...")
    model = load_flowfeat(device=device)

    print(f"Loading video: {VIDEO_PATH}")
    pixels = load_video_frames(VIDEO_PATH).to(device)
    
    print("getting FlowFeat features...")
    feats = get_flowfeat(pixels, model)     #[1, T, 128, 224, 224]
    
    #now, clustering needs to happen
    Ps = sample_semantic_points(feats, M=85, clusters=30)   # [256, 2]  (x, y) coords
    
    _, T, C, H, W = feats.shape
    # visualize sampled points on first frame
    raw = cv2.imread  # don't do this, use VideoCapture instead
    cap = cv2.VideoCapture(VIDEO_PATH)
    _, raw_frame = cap.read()   # already BGR, already [0,255]
    cap.release()
    raw_frame = cv2.resize(raw_frame, (W, H))   # match the resolution flowfeat used

    for x, y in Ps.int().tolist():
        cv2.circle(raw_frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(f'point_sampling/{VIDEO_NAME}_sampled_points.png', raw_frame)
    print(f"saved visualization to {VIDEO_NAME}_sampled_points.png")