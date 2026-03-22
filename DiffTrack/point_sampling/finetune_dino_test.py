import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from model_finetune import DINOv2_LoRA, get_robust_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import v2
import sys
from depth_anything_3.api import DepthAnything3

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from utils.depth_exp import get_batch_depth

checkpoint_path = "test_models/physics_guide_lora_dino_epoch_9.pth" 
video_path = "UCF_Rep/val/v_BenchPress_g22_c01.mp4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#raft transforms for the flow teacher
raft_transform = v2.Compose([
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    v2.Resize(size=(520, 960)),
])

def visualize():
    #1. init all models
    print("loading models onto gpu...")
    model = DINOv2_LoRA().to(device)
    
    #load lora weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    #init flow teacher (raft)
    raft_model = raft_large(pretrained=True, progress=False).to(device).eval()
    
    #init depth teacher
    # print("loading depth model...")
    # depth_model = DepthAnything3.from_pretrained("depth-anything/da3-base").to(device).eval()
    
    #2. extract frames (WITH STRIDE FIX)
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    pixel_frames = []
    
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    start_f = 2 #pick a frame in the middle
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f * 2) # Adjust start frame due to stride
    
    for _ in range(3): 
        ret, frame = cap.read()
        if not ret: break
        
        # FIX 1: read and discard the next frame to match stride=2
        cap.read() 
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (224, 224))
        raw_frames.append(resized)
        
        #pixel input for dino/seghead
        t_frame = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        pixel_frames.append((t_frame.to(device) - norm_mean) / norm_std)
    cap.release()

    #3. calculate depth on the fly
    # print("generating depth...")
    #using your utility function - target size should match pixel size
    # depth_tensor = get_batch_depth(raw_frames, depth_model, target_size=(224, 224)) # [3, 1, 224, 224]
    # depth_tensor = depth_tensor.to(device)

    #4. calculate flow on the fly
    print("generating optical flow...")
    flow_list = []
    for i in range(2):
        img1 = raft_transform(torch.from_numpy(raw_frames[i]).permute(2,0,1)).to(device).unsqueeze(0)
        img2 = raft_transform(torch.from_numpy(raw_frames[i+1]).permute(2,0,1)).to(device).unsqueeze(0)
        
        with torch.no_grad():
            list_of_flows = raft_model(img1, img2)
            #interpolate back to 224x224
            flow_res = F.interpolate(list_of_flows[-1], size=(224, 224), mode="bilinear")
            flow_list.append(flow_res.squeeze(0))
            
    # FIX 2: duplicate the last flow instead of zero-padding to keep the min() filter alive
    flow_list.append(flow_list[-1].clone())
    flow_tensor = torch.stack(flow_list) # [3, 2, 224, 224]

    #5. inference
    print("running inference...")
    input_pixels = torch.stack(pixel_frames).unsqueeze(0) # [1, 3, 3, 224, 224]
    
    with torch.no_grad():
        outputs = model(input_pixels)
        pred_mask = outputs["pred_mask"]
        
        #generate physics mask using the on-the-spot flow/depth
        teacher_mask = get_robust_mask(flow_tensor.unsqueeze(0))

    #6. plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    idx = 1 #show the middle frame of the chunk
    
    axes[0].imshow(raw_frames[idx])
    axes[0].set_title("original frame")
    axes[0].axis('off')

    #physics teacher
    t_map = teacher_mask[0, idx, 0].cpu().numpy()
    axes[1].imshow(raw_frames[idx])
    axes[1].imshow(t_map, cmap='jet', alpha=0.5)
    axes[1].set_title("on-the-fly physics teacher")
    axes[1].axis('off')

    #student
    s_map = F.interpolate(pred_mask[0, idx].unsqueeze(0), size=(224, 224), mode='bilinear').squeeze().cpu().numpy()
    axes[2].imshow(raw_frames[idx])
    axes[2].imshow(s_map, cmap='jet', alpha=0.5)
    axes[2].set_title("student (seg head)")
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs("point_sampling/finetuned_test_new/", exist_ok=True)
    save_name = f"point_sampling/finetuned_test_new/{os.path.basename(video_path)}_live_test.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"saved results to {save_name}")
    
def visualize_physics_mask(video_path, start_frame=0, device="cuda"):
    # 1. Initialize Teacher Models
    print("Initializing teachers...")
    raft_model = raft_large(pretrained=True, progress=False).to(device).eval()
    
    raft_transform = v2.Compose([
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        v2.Resize(size=(520, 960)),
    ])

    # 2. Extract 3 Frames with Stride=2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame * 2)
    
    raw_frames = []
    for _ in range(3):
        ret, frame = cap.read()
        if not ret: break
        cap.read() # Discard next frame for stride=2
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(cv2.resize(frame_rgb, (224, 224)))
    cap.release()

    if len(raw_frames) < 3:
        print("Error: Could not extract 3 frames.")
        return

    # 3. Generate Depth and Flow
    print("Generating flow...")
    with torch.no_grad():
        # Depth
        # depth_tensor = get_batch_depth(raw_frames, depth_model, target_size=(224, 224)).to(device)
        
        # Flow
        flow_list = []
        for i in range(2):
            img1 = raft_transform(torch.from_numpy(raw_frames[i]).permute(2,0,1)).to(device).unsqueeze(0)
            img2 = raft_transform(torch.from_numpy(raw_frames[i+1]).permute(2,0,1)).to(device).unsqueeze(0)
            
            flows = raft_model(img1, img2)
            res_flow = F.interpolate(flows[-1], size=(224, 224), mode="bilinear")
            flow_list.append(res_flow.squeeze(0))
        
        # Mirror last flow to maintain temporal dimension for min() filter
        flow_list.append(flow_list[-1].clone())
        flow_tensor = torch.stack(flow_list).unsqueeze(0) # [1, 3, 2, 224, 224]

    # 4. Compute Mask
    print("Computing mask...")
    # Unsqueeze depth to [1, 3, 1, 224, 224] to match get_robust_mask expectations
    mask = get_robust_mask(flow_tensor)
    
    # 5. Visualize + Save
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Replace single-frame view with all 3 frames, then return early
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    for idx in range(3):
        axes[idx, 0].imshow(raw_frames[idx])
        axes[idx, 0].set_title(f"Original Frame {idx}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(raw_frames[idx])
        axes[idx, 1].imshow(mask[0, idx, 0].cpu().numpy(), cmap="jet", alpha=0.5)
        axes[idx, 1].set_title(f"Physics Teacher Mask {idx}")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    os.makedirs("point_sampling/physics_mask_vis", exist_ok=True)
    save_path = os.path.join(
        "point_sampling/physics_mask_vis",
        f"{os.path.basename(video_path)}_start{start_frame}_physics_mask.png",
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")
    return

    
if __name__ == "__main__":
    visualize()
    # visualize_physics_mask("UCF_Rep/val/v_FrontCrawl_g22_c01.mp4", start_frame=20)
    # visualize_physics_mask("vids_mp4/swim_2.mp4", start_frame=5)