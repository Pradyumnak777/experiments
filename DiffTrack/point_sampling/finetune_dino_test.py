import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from model_finetune import DINOv2_LoRA
import matplotlib.pyplot as plt
from transformers import AutoModel
# from sklearn.cluster import KMeans

checkpoint_path = "test_models/new_lora_dino_epoch_49.pth" 
video_path = "UCF_Rep/val/v_CuttingInKitchen_g21_c01.mp4" #pick a specific video to test
# video_path = "vids_mp4/benchpress.mp4" #pick a specific video to test 
# video_path = "countix/-dxBq0WzYRU_35.42309_37.71705.mp4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize():
    model = DINOv2_LoRA().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded weights from {checkpoint_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    raw_frames = []
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    for _ in range(1): #just look at the first frame
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

    with torch.no_grad():
        #features shape: [1, T, 768, 16, 16]
        outputs = model(input_tensor)
        
        #calculate self-similarity to find the 'actor'
        #we take the average feature vector as the 'goal'
        #then see which patches look most like it
        features = outputs["patch_features"]
        predicted_mask = outputs["pred_mask"]

        # '''
        # alt method
        # '''
        # corners = torch.stack([
        #     features[:, :, :, 0, 0],   # Top-left
        #     features[:, :, :, 0, -1],  # Top-right
        #     features[:, :, :, -1, 0],  # Bottom-left
        #     features[:, :, :, -1, -1]  # Bottom-right
        # ], dim=-1) # Shape: [1, T, 768, 4]

        # # Average ONLY these corner patches to get a 100% pure "Background Vector"
        # pure_bg_feature = corners.mean(dim=(1, 3), keepdim=True).unsqueeze(-1) # [1, 1, 768, 1, 1]
        
        # # Background will now perfectly match (Red). Actor will heavily mismatch (Deep Blue).
        # sim_map = torch.cosine_similarity(features, pure_bg_feature, dim=2)
    
    # with torch.no_grad():
    #     features = model(input_tensor) # [1, T, 768, 16, 16]
    #     _, T, C, H, W = features.shape
        
    #     # 1. Flatten all patches across space and time into a 2D matrix
    #     # Shape becomes [T * H * W, 768] (e.g., [2048, 768])
    #     features_flat = features.view(T, C, H * W).permute(0, 2, 1).reshape(T * H * W, C)
        
    #     # 2. Mean-center the features (required for PCA)
    #     features_flat = features_flat - features_flat.mean(dim=0, keepdim=True)
        
    #     # 3. Run PCA to find the top 3 principal components
    #     # U contains the projected scores for each patch
    #     U, S, V = torch.pca_lowrank(features_flat, q=3)
        
    #     # 4. Grab the First Principal Component (PC1)
    #     # PC1 represents the axis of greatest variance (Foreground vs Background)
    #     pc1 = U[:, 0].view(1, T, H, W)
        
    #     # 5. Auto-Invert logic
    #     # PCA directions are arbitrary (+ or -). We assume corners are background.
    #     # If the corners have a positive score, we flip the sign so the actor is positive.
    #     corners_mean = (pc1[0, 0, 0, 0] + pc1[0, 0, 0, -1] + pc1[0, 0, -1, 0] + pc1[0, 0, -1, -1]) / 4.0
    #     if corners_mean > 0:
    #         pc1 = -pc1
            
    #     # Rename back to sim_map so your plotting code works without changes
    #     sim_map = pc1
        
    #4. plotting (first frame only)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    heatmap_tensor = predicted_mask[0, 0].unsqueeze(0)

    heatmap = F.interpolate(
        heatmap_tensor, # Take first frame of the pair
        size=(224, 224),
        mode='bilinear'
    ).squeeze().cpu().numpy()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    axes[0].imshow(raw_frames[0])
    axes[0].set_title("Frame 0 (Original)")
    axes[0].axis('off')

    axes[1].imshow(raw_frames[0])
    im = axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[1].set_title("Frame 0 (Mask Overlay)")
    axes[1].axis('off')

    # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    # cbar.set_label('similarity to actor vector', rotation=270, labelpad=15)

    plt.tight_layout()
    os.makedirs("point_sampling/finetuned_test_new/", exist_ok=True)
    
    # Extract video name without .mp4 and leading directories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_name = f"point_sampling/finetuned_test_new/{video_name}_dino_mask.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"saved visualization to {save_name}")


def visualize_plain():
    model_name = 'facebook/dinov2-base'
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Loaded vanilla {model_name} from HuggingFace")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    raw_frames = []
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    for _ in range(1): #just look at the first frame
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

    with torch.no_grad():
        B, T, C, H, W = input_tensor.shape
        x = input_tensor.view(B * T, C, H, W) 
        
        outputs = model(x)
        features_raw = outputs.last_hidden_state # [B*T, 257, 768]
        
        patch_features = features_raw[:, 1:, :].permute(0, 2, 1).view(B * T, 768, 16, 16)
        features = patch_features.view(B, T, 768, 16, 16)
        
        avg_feature = features.mean(dim=(1, 3, 4), keepdim=True) 
        sim_map = torch.cosine_similarity(features, avg_feature, dim=2)
    
    # with torch.no_grad():
    #     features = model(input_tensor) # [1, T, 768, 16, 16]
    #     _, T, C, H, W = features.shape
        
    #     # 1. Flatten all patches across space and time into a 2D matrix
    #     # Shape becomes [T * H * W, 768] (e.g., [2048, 768])
    #     features_flat = features.view(T, C, H * W).permute(0, 2, 1).reshape(T * H * W, C)
        
    #     # 2. Mean-center the features (required for PCA)
    #     features_flat = features_flat - features_flat.mean(dim=0, keepdim=True)
        
    #     # 3. Run PCA to find the top 3 principal components
    #     # U contains the projected scores for each patch
    #     U, S, V = torch.pca_lowrank(features_flat, q=3)
        
    #     # 4. Grab the First Principal Component (PC1)
    #     # PC1 represents the axis of greatest variance (Foreground vs Background)
    #     pc1 = U[:, 0].view(1, T, H, W)
        
    #     # 5. Auto-Invert logic
    #     # PCA directions are arbitrary (+ or -). We assume corners are background.
    #     # If the corners have a positive score, we flip the sign so the actor is positive.
    #     corners_mean = (pc1[0, 0, 0, 0] + pc1[0, 0, 0, -1] + pc1[0, 0, -1, 0] + pc1[0, 0, -1, -1]) / 4.0
    #     if corners_mean > 0:
    #         pc1 = -pc1
            
    #     # Rename back to sim_map so your plotting code works without changes
    #     sim_map = pc1
        
    #4. plotting (first frame only)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    heatmap = F.interpolate(
        sim_map[0, 0].view(1, 1, 16, 16),
        size=(224, 224),
        mode='bilinear'
    ).squeeze().cpu().numpy()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    axes[0].imshow(raw_frames[0])
    axes[0].set_title("Frame 0 (Original)")
    axes[0].axis('off')

    axes[1].imshow(raw_frames[0])
    im = axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[1].set_title("Frame 0 (Mask Overlay)")
    axes[1].axis('off')

    # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    # cbar.set_label('similarity to actor vector', rotation=270, labelpad=15)

    plt.tight_layout()
    os.makedirs("point_sampling/finetuned_test/", exist_ok=True)
    
    # Extract video name without .mp4 and leading directories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_name = f"point_sampling/finetuned_test/{video_name}_plain_dino_mask.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"saved visualization to {save_name}")

if __name__ == "__main__":
    visualize()
    # visualize_plain()