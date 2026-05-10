import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from model_finetune import DINOv2_LoRA
import matplotlib.pyplot as plt

checkpoint_path = "test_models/lora_dino_epoch_0.pth" 
# video_path = "UCF_Rep/val/v_PlayingViolin_g22_c02.mp4" #pick a specific video to test
video_path = "vids_mp4/74xHYgPwErQ_4.0_8.72.mp4" #pick a specific video to test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_maskcut(frame_features):
    #frame_features shape: [768, 16, 16]
    c, h, w = frame_features.shape
    
    #flatten to [256, 768] and normalize
    F_flat = frame_features.view(c, -1).t()
    F_flat = F.normalize(F_flat, p=2, dim=1)
    
    #1. build the affinity matrix (cosine similarity between all pairs of patches)
    W = torch.mm(F_flat, F_flat.t()) # [256, 256]
    
    #threshold to keep only positive connections (removes noise)
    W = torch.clamp(W, min=0)
    
    #2. compute symmetric normalized laplacian/affinity
    D = W.sum(dim=1)
    D_inv_sqrt = 1.0 / torch.sqrt(D + 1e-8)
    W_norm = D_inv_sqrt.unsqueeze(1) * W * D_inv_sqrt.unsqueeze(0)
    
    #enforce symmetry for numerical stability before eigendecomposition
    W_norm = (W_norm + W_norm.t()) / 2.0
    
    #3. solve for eigenvectors
    #eigh returns eigenvalues in ascending order
    evals, evecs = torch.linalg.eigh(W_norm)
    
    #the largest eigenvalue [-1] is the whole graph. 
    #the second largest [-2] is the primary cut (foreground vs background)
    eigenvector = evecs[:, -2]
    
    #reshape back to spatial grid
    mask = eigenvector.view(h, w)
    
    # #4. auto-inversion logic
    # #we assume the 4 corners of the frame are background (gym floor, pool, room walls)
    # #if the eigenvector assigned positive values to the corners, it means the background is 'hot'.
    # #so we flip it, guaranteeing the actor becomes the positive peak!
    # corners_mean = (mask[0, 0] + mask[0, -1] + mask[-1, 0] + mask[-1, -1]) / 4.0
    # if corners_mean > 0:
    #     mask = -mask
        
    return mask

def visualize():
    #1. init model and load lora weights
    model = DINOv2_LoRA().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded weights from {checkpoint_path}")

    #2. grab a few frames from the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    raw_frames = []
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    for _ in range(8): #just look at the first 8 frames
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(cv2.resize(frame_rgb, (224, 224)))
        
        t_frame = torch.from_numpy(raw_frames[-1]).permute(2, 0, 1).float() / 255.0
        t_frame = t_frame.to(device) #move to gpu first!
        t_frame = (t_frame - norm_mean) / norm_std
        frames.append(t_frame)
    cap.release()

    #stack and move to device [1, T, 3, 224, 224]
    input_tensor = torch.stack(frames).unsqueeze(0).to(device)

    #3. run inference
    with torch.no_grad():
        #features shape: [1, T, 768, 16, 16]
        features = model(input_tensor)
        
        #run maskcut per frame and stack the results
        maskcut_maps = []
        for t in range(features.shape[1]):
            frame_mask = get_maskcut(features[0, t])
            maskcut_maps.append(frame_mask)
            
        #stack to [T, 16, 16]
        sim_map = torch.stack(maskcut_maps) 
        
    #4. plotting
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    
    im = None 
    for i in range(min(8, len(raw_frames))):
        #upscale 16x16 maskcut map back to 224x224 for viewing
        heatmap = F.interpolate(
            sim_map[i].view(1, 1, 16, 16), 
            size=(224, 224), 
            mode='bilinear'
        ).squeeze().cpu().numpy()
        
        #normalize heatmap for visualization so 0 is background and 1 is actor
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        #overlay logic
        axes[i].imshow(raw_frames[i])
        im = axes[i].imshow(heatmap, cmap='jet', alpha=0.5) #alpha 0.5 let's us see the actor underneath
        axes[i].set_title(f"Frame {i}")
        axes[i].axis('off')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label('maskcut eigenvector score', rotation=270, labelpad=15)

    plt.tight_layout()
    os.makedirs("point_sampling/finetuned_test/", exist_ok=True)
    
    # Extract video name without .mp4 and leading directories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_name = f"point_sampling/finetuned_test/{video_name}_dino_maskcut.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"saved visualization to {save_name}")

if __name__ == "__main__":
    visualize()