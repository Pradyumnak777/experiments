import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os

# ── paths (run from DiffTrack/) ───────────────────────────────────────────────
VIDEO_NAME  = 'v_Biking_g21_c01'
VIDEO_PATH  = f'UCF_Rep/val/{VIDEO_NAME}.mp4'
OUTPUT_PATH = 'point_sampling/flowfeat_mask.png'

NUM_FRAMES  = 8
PERCENTILE  = 60
SIZE        = (224, 224)
# ─────────────────────────────────────────────────────────────────────────────


def load_flowfeat(model_name='dinov2_vitb14_kt', device='cuda'):
    model = torch.hub.load('tum-vision/flowfeat', 'flowfeat', name=model_name, pretrained=True)
    return model.to(device).eval()

@torch.no_grad()
def get_flowfeat_mask(pixels, flowfeat_model):
    B, T, C, H, W = pixels.shape
    device = pixels.device

    frames_flat = pixels.view(B * T, C, H, W)
    outputs = flowfeat_model(frames_flat)

    # VERIFY: output format — dict or tuple — from the repo demo notebook
    if isinstance(outputs, dict):
        decoder_feats = outputs['decoder']
    else:
        decoder_feats = outputs[1]          # assume (encoder, decoder)

    _, feat_dim, fH, fW = decoder_feats.shape
    if fH != H or fW != W:
        decoder_feats = F.interpolate(
            decoder_feats, size=(H, W), mode='bilinear', align_corners=False
        )

    decoder_feats = F.normalize(decoder_feats, dim=1)
    decoder_feats = decoder_feats.view(B, T, feat_dim, H, W)

    mean_feat = decoder_feats.mean(dim=1, keepdim=True)
    variance  = ((decoder_feats - mean_feat) ** 2).mean(dim=(1, 2))  # [B, H, W]

    thresh = torch.quantile(
        variance.view(B, -1), PERCENTILE / 100.0, dim=1
    ).view(B, 1, 1)

    binary_mask = (variance > thresh).float()
    return binary_mask.unsqueeze(1).unsqueeze(2).expand(-1, T, 1, -1, -1).contiguous()


def load_video_frames(mp4_path):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    cap   = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, SIZE)
        frame = (frame / 255.0 - mean) / std
        frames.append(frame)
    cap.release()

    tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    return tensor.unsqueeze(0)   # [1, T, 3, H, W]


def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    return np.clip(img * std + mean, 0, 1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading FlowFeat...")
    model = load_flowfeat(device=device)

    print(f"Loading video: {VIDEO_PATH}")
    pixels = load_video_frames(VIDEO_PATH).to(device)

    print("Computing mask...")
    mask = get_flowfeat_mask(pixels, model)   # [1, T, 1, 224, 224]

    T = pixels.shape[1]
    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))

    for t in range(T):
        img = denorm(pixels[0, t])
        msk = mask[0, t, 0].cpu().numpy()

        axes[0, t].imshow(img)
        axes[0, t].set_title(f'frame {t}')
        axes[0, t].axis('off')

        overlay = img.copy()
        overlay[msk == 1] = overlay[msk == 1] * 0.4 + np.array([1, 0, 0]) * 0.6
        axes[1, t].imshow(overlay)
        axes[1, t].set_title(f'mask {t}')
        axes[1, t].axis('off')

    plt.suptitle(f'{VIDEO_NAME}  |  FlowFeat mask  (percentile={PERCENTILE})')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved → {OUTPUT_PATH}")