import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from model_finetune import DINOv2_LoRA
import matplotlib.pyplot as plt

checkpoint_path = "test_models/lora_dino_epoch_0.pth" 
# video_path = "UCF_Rep/val/v_Rowing_g21_c03.mp4" #pick a specific video to test
video_path = "vids_mp4/9wyq-dzy2TI_22.0_26.666667.mp4" #pick a specific video to test 
# video_path = "countix/-dxBq0WzYRU_35.42309_37.71705.mp4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        #calculate self-similarity to find the 'actor'
        #we take the average feature vector as the 'goal'
        #then see which patches look most like it
        avg_feature = features.mean(dim=(1, 3, 4), keepdim=True) # [1, 1, 768, 1, 1]
        sim_map = torch.cosine_similarity(features, avg_feature, dim=2) # [1, T, 16, 16]
        
    #4. plotting
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    
    im = None 
    for i in range(min(8, len(raw_frames))):
        #upscale 16x16 similarity map back to 224x224 for viewing
        heatmap = F.interpolate(
            sim_map[0, i].view(1, 1, 16, 16), 
            size=(224, 224), 
            mode='bilinear'
        ).squeeze().cpu().numpy()
        
        #normalize heatmap for visualization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        #overlay logic
        axes[i].imshow(raw_frames[i])
        im = axes[i].imshow(heatmap, cmap='jet', alpha=0.5) #alpha 0.5 let's us see the actor underneath
        axes[i].set_title(f"Frame {i}")
        axes[i].axis('off')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label('similarity to actor vector', rotation=270, labelpad=15)

    plt.tight_layout()
    os.makedirs("point_sampling/finetuned_test/", exist_ok=True)
    
    # Extract video name without .mp4 and leading directories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_name = f"point_sampling/finetuned_test/{video_name}_dino_mask.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"saved visualization to {save_name}")

if __name__ == "__main__":
    visualize()