'''
trokens-style point sampling, using DINO (not flowfeat)
1. take input video
2. perform DINO to get 768 dim patch tokens (16x16 grid per frame)
3. cluster the patch tokens, sample points -> map patch back to pixel coords
'''

import cv2
from dino_pca import load_dino, get_dino_feats   #dino loader + [B, 256, 768] patch tokens
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import numpy as np

VIDEO_NAME  = 'v_CuttingInKitchen_g21_c01'
VIDEO_PATH  = f'UCF_Rep/val/{VIDEO_NAME}.mp4'
SIZE        = (224, 224)

def sample_semantic_points(patch_tokens, M, clusters, grid=16, img_size=224, seed=0):
    np.random.seed(seed)
    B, num_patches, C = patch_tokens.shape   #[B, 256, 768]
    q = M // clusters                        # points per cluster
    patch_px = img_size // grid              # pixels per patch (14 for dinov2-base)

    # take first frame patch tokens [256, 768]
    flat = patch_tokens[0].detach().float()                  # [256, 768]
    flat = F.normalize(flat, dim=-1).cpu().numpy()

    # cluster patch tokens into semantic groups (same as trokens)
    labels = MiniBatchKMeans(n_clusters=clusters, random_state=seed).fit_predict(flat)  # [256]

    # sample q patches per cluster, map patch -> pixel center
    sampled = []
    for cid in range(clusters):
        idx    = np.where(labels == cid)[0]
        chosen = np.random.choice(idx, size=min(q, len(idx)), replace=False)
        rows, cols = chosen // grid, chosen % grid           # patch grid coords
        xs = cols * patch_px + patch_px // 2                  # pixel center x
        ys = rows * patch_px + patch_px // 2                  # pixel center y
        sampled.extend(zip(xs.tolist(), ys.tolist()))        # (x, y) = (col, row)

    return torch.tensor(sampled[:M], dtype=torch.float32)    # [M, 2]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading DINO...")
    model, device = load_dino('facebook/dinov2-base', device=device)

    print(f"Loading video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()   # first frame, BGR
    cap.release()
    assert ret, f"failed to read {VIDEO_PATH}"

    # squash to 224x224 (no center crop) so patch coords line up with the viz
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(SIZE)

    print("getting DINO features...")
    patch_tokens = get_dino_feats(model, image, device)   #[1, 256, 768]

    #now, clustering needs to happen
    Ps = sample_semantic_points(patch_tokens, M=85, clusters=30)   # [M, 2]  (x, y) coords

    # visualize sampled points on first frame
    raw_frame = cv2.resize(frame, SIZE)   # 224x224, BGR, [0,255]
    for x, y in Ps.int().tolist():
        cv2.circle(raw_frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(f'point_sampling/{VIDEO_NAME}_sampled_points_dino.png', raw_frame)
    print(f"saved visualization to {VIDEO_NAME}_sampled_points_dino.png")