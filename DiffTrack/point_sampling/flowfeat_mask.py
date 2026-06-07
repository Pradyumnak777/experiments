import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') #not use gui
import matplotlib.pyplot as plt
import cv2

VIDEO_NAME  = 'v_Biking_g21_c01'
VIDEO_PATH  = f'UCF_Rep/val/{VIDEO_NAME}.mp4'
OUTPUT_PATH = 'point_sampling/flowfeat_pca.png'

NUM_FRAMES  = 1
SIZE        = (224, 224)


def load_flowfeat(model_name='dinov2_vitb14_kt', device='cuda'):
    model = torch.hub.load('tum-vision/flowfeat', 'flowfeat', name=model_name, pretrained=True)
    return model.to(device).eval()


@torch.no_grad()
def get_flowfeat(pixels, flowfeat_model):
    """Return the raw FlowFeat decoder features, upsampled to input size.

    Output shape: [B, T, feat_dim, H, W]
    """
    B, T, C, H, W = pixels.shape

    frames_flat = pixels.view(B * T, C, H, W)
    outputs = flowfeat_model(frames_flat)

    if isinstance(outputs, dict):
        decoder_feats = outputs['decoder']
    else:
        decoder_feats = outputs[1]          # assume (encoder, decoder)

    _, feat_dim, fH, fW = decoder_feats.shape
    if fH != H or fW != W:
        decoder_feats = F.interpolate(
            decoder_feats, size=(H, W), mode='bilinear', align_corners=False
        )

    return decoder_feats.view(B, T, feat_dim, H, W)


@torch.no_grad()
def feats_to_rgb(decoder_feats):
    """PCA the FlowFeat channels down to 3 -> RGB, jointly over all frames.

    Fitting one PCA over the whole clip keeps the colour mapping consistent
    across frames, so temporal consistency is visible.

    Input : [B, T, C, H, W]   (assumes B == 1)
    Output: [T, H, W, 3] in [0, 1]
    """
    B, T, C, H, W = decoder_feats.shape
    assert B == 1, "visualization assumes batch size 1"

    # [T*H*W, C]
    feats = decoder_feats[0].permute(0, 2, 3, 1).reshape(-1, C).float()

    # center, then project onto top-3 principal components
    mean = feats.mean(dim=0, keepdim=True)
    centered = feats - mean
    _, _, V = torch.pca_lowrank(centered, q=3)
    proj = centered @ V[:, :3]              # [T*H*W, 3]

    # min-max normalize each component over the whole volume -> [0, 1]
    lo = proj.min(dim=0, keepdim=True).values
    hi = proj.max(dim=0, keepdim=True).values
    rgb = (proj - lo) / (hi - lo + 1e-8)

    return rgb.reshape(T, H, W, 3).cpu().numpy()


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

    print("Extracting FlowFeat features...")
    feats = get_flowfeat(pixels, model)     # [1, T, C, 224, 224]
    rgb   = feats_to_rgb(feats)             # [T, 224, 224, 3]

    T = pixels.shape[1]
    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))
    if NUM_FRAMES == 1:
        axes = axes[:, np.newaxis]

    for t in range(T):
        axes[0, t].imshow(denorm(pixels[0, t]))
        axes[0, t].set_title(f'frame {t}')
        axes[0, t].axis('off')

        axes[1, t].imshow(rgb[t])
        axes[1, t].set_title(f'flowfeat {t}')
        axes[1, t].axis('off')

    plt.suptitle(f'{VIDEO_NAME}  |  raw FlowFeat (PCA->RGB, shared across frames)')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved → {OUTPUT_PATH}")