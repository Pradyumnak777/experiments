import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from model_finetune import DINOv2_LoRA
import matplotlib.pyplot as plt

checkpoint_path = "test_models/newv2_lora_dino_epoch_24.pth" 
video_path = "UCF_Rep/val/v_JumpRope_g25_c02.mp4" #pick a specific video to test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize():
    model = DINOv2_LoRA().to(device)
    
    # load weights (accounting for potential dataparallel saving)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # if it complains about "module." prefixes, let me know!
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded weights from {checkpoint_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    raw_frames = []
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # the model expects T=2, so we grab the first 2 frames
    for _ in range(2): 
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(cv2.resize(frame_rgb, (224, 224)))
        
        t_frame = torch.from_numpy(raw_frames[-1]).permute(2, 0, 1).float() / 255.0
        t_frame = t_frame.to(device) 
        t_frame = (t_frame - norm_mean) / norm_std
        frames.append(t_frame)
    cap.release()

    # stack and move to device [1, 2, 3, 224, 224]
    input_tensor = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        # run inference! 
        outputs = model(input_tensor)
        
        # we don't need PCA or similarity maps anymore. 
        # the model directly predicts the actor probability!
        predicted_mask = outputs["pred_mask"] # shape: [1, 2, 1, 16, 16]

    # 4. plotting (first frame only)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # grab the mask for batch 0, time 0
    # shape becomes [1, 1, 16, 16] so interpolate can handle it
    heatmap_tensor = predicted_mask[0, 0].unsqueeze(0) 

    # upsample to 224x224 smoothly
    heatmap = F.interpolate(
        heatmap_tensor,
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()

    # because it went through a Sigmoid, values are already roughly 0 to 1
    # but we can normalize it just to make the heatmap colors pop perfectly
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    axes[0].imshow(raw_frames[0])
    axes[0].set_title("Frame 0 (Original)")
    axes[0].axis('off')

    axes[1].imshow(raw_frames[0])
    im = axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[1].set_title("Frame 0 (Predicted Seg Head)")
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs("point_sampling/finetuned_test_new/", exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_name = f"point_sampling/finetuned_test_new/{video_name}_dino_mask.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"saved visualization to {save_name}")

if __name__ == "__main__":
    visualize()