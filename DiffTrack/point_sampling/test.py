import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from transformers import AutoModel
from torchvision.models.optical_flow import raft_large

from depth_anything_3.api import DepthAnything3
from model_2 import MaskGen
from data_utils import mp4_to_frames, tensorize_vid, transform, preprocess

def test_single_video(video_path, checkpoint_path):
    device = torch.device("cuda:4" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading feature extraction models...")
    depth_model = DepthAnything3.from_pretrained("depth-anything/da3-base").to(device)
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base', output_attentions=True).to(device).eval()
    raft_model = raft_large(pretrained=True, progress=False).to(device).eval()

    print(f"\nProcessing test video: {video_path}")
    vid_name = os.path.basename(video_path)
    frames = mp4_to_frames(video_path)
    video_tensor = tensorize_vid(frames, transform)
    
    # It will create the folder and save the .pt files if they don't exist yet.
    preprocess(video_tensor, frames, vid_name, dino_model, raft_model, depth_model, stride=2)
    
    print("\nLoading trained MaskGen weights...")
    model = MaskGen().to(device)
    
    # Load the weights (state_dict) you saved
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # Set to evaluation mode! Very important.

    # ---------------------------------------------------------
    # 4. PREPARE THE 5D TENSOR FOR INFERENCE
    # ---------------------------------------------------------
    print("Preparing tensor and running inference...")
    v_path = os.path.join("ucfrep_intermediate_dataset", os.path.splitext(vid_name)[0])
    
    # Grab the first 8 frames (a single clip) to test the model
    # Note: Because stride=2 during preprocessing, these 8 sparse frames represent 16 actual frames
    dino = torch.load(os.path.join(v_path, "dino.pt"), weights_only=True)[0:8].float()
    flow = torch.load(os.path.join(v_path, "flow.pt"), weights_only=True)[0:8].float()
    depth = torch.load(os.path.join(v_path, "depth.pt"), weights_only=True)[0:8].float()
    
    # Interpolate DINO up to 224x224
    dino = F.interpolate(dino, size=(224, 224), mode='bilinear', align_corners=False) 
    
    # Stack features: [8(Time), 387(Channels), 224(H), 224(W)]
    clip = torch.cat([dino, flow, depth], dim=1)
    
    # Add Batch dimension and permute for Conv3d: [1(Batch), 387(Channels), 8(Time), 224(H), 224(W)]
    clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
    
    with torch.no_grad():
        mask_pred, _ = model(clip) # Output shape: [1, 1, 224, 224]
        
    print("Generating visualization...")
    # Squeeze out the batch and channel dimensions to get a [224, 224] 2D array
    mask_np = mask_pred.squeeze().cpu().numpy()
    
    # Grab the raw RGB frame corresponding to t=0 (first frame of the clip)
    # We resize it to 224x224 to match the mask output resolution
    orig_frame = cv2.resize(frames[0], (224, 224))
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original Frame
    plt.subplot(1, 3, 1)
    plt.title("Original Frame (t=0)")
    plt.imshow(orig_frame)
    plt.axis('off')
    
    # Plot 2: Predicted Mask
    plt.subplot(1, 3, 2)
    plt.title("Predicted Probability Mask")
    # 'viridis' or 'jet' are great colormaps for seeing probabilities
    plt.imshow(mask_np, cmap='viridis') 
    plt.colorbar(fraction=0.046, pad=0.04) # Adds a scale to show the 0.0 to 1.0 confidence
    plt.axis('off')
    
    # Plot 3: Overlay (The "Truth")
    plt.subplot(1, 3, 3)
    plt.title("Mask Overlay")
    plt.imshow(orig_frame)
    # Alpha=0.6 makes the mask semi-transparent so you can see the actor underneath
    plt.imshow(mask_np, cmap='viridis', alpha=0.6) 
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the result
    save_dir = "test_imgs"
    os.makedirs(save_dir, exist_ok=True)

    save_filename = os.path.join(save_dir, f"inference_result_{os.path.splitext(vid_name)[0]}.png")
    plt.savefig(save_filename, dpi=300)
    print(f"Success! Visualization saved as: {save_filename}")

if __name__ == "__main__":
    # --- UPDATE THIS PATH ---
    test_mp4_path = "UCF_Rep/val/v_SoccerJuggling_g22_c03.mp4" 
    # test_mp4_path = "vids_mp4/swim_2.mp4" 

    # weights_path = "test_models/curr_use_dinov2_epoch2.pth"
    weights_path = "test_models/mask_gen_epoch_2.pth"

    
    if os.path.exists(test_mp4_path) and os.path.exists(weights_path):
        test_single_video(test_mp4_path, weights_path)
    else:
        print("Error: Please check the paths to your test .mp4 and .pth files!")