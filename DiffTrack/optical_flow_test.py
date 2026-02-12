import numpy as np
import torch
import torchvision.transforms as T
import glob
import os
from torchvision.io import read_image
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import torchvision


def read_frames_as_video(folder_path, extension="*.jpg"):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, extension)))
    
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {folder_path} with extension {extension}")
    
    print(f"Found {len(frame_paths)} frames in {folder_path}")
    
    frame_tensors = [] #(T, C, H, W)
    for frame_path in frame_paths:
        img = read_image(frame_path) #converts to tensor too 
        frame_tensors.append(img)

    video_tensor = torch.stack(frame_tensors, 0) #stack along temporal dim
    
    return video_tensor

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch


name = "swim_3"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use absolute path or change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
video_folder = os.path.join(script_dir, "videos/swim_3")

frames = read_frames_as_video(video_folder)
img1_raw = frames[:-1] 
img2_raw = frames[1:] 


model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

# inference loop
batch_size = 2  # Process 2 pairs at a time to save memory
predicted_flows_list = []

print(f"Processing {len(img1_raw)} pairs...")

with torch.no_grad():
    for i in range(0, len(img1_raw), batch_size):
        # Slice mini-batches
        img1_batch = img1_raw[i : i + batch_size]
        img2_batch = img2_raw[i : i + batch_size]
        
        # Preprocess and move ONLY this batch to GPU
        img1_batch = preprocess(img1_batch).to(device)
        img2_batch = preprocess(img2_batch).to(device)

        # Run model
        list_of_flows = model(img1_batch, img2_batch)
        
        # Save result to CPU list
        predicted_flows_list.append(list_of_flows[-1].cpu())

# Concatenate all mini-batches back into one big tensor
predicted_flows = torch.cat(predicted_flows_list, dim=0)

print(f"Flow generation complete. Shape: {predicted_flows.shape}")


flow_imgs = flow_to_image(predicted_flows)

resize_transform = T.Resize(size=(520, 960))
img1_batch_viz = resize_transform(img1_raw).float() / 255.0 # Normalize to [0, 1]

# optical flow save as video
output_dir = "thesis/DiffTrack/output_optical_flow"
os.makedirs(output_dir, exist_ok=True)

# option 1: flow visualization
flow_video_path = os.path.join(output_dir, f"{name}_optical_flow.mp4")
torchvision.io.write_video(flow_video_path, flow_imgs.permute(0, 2, 3, 1).cpu(), fps=20)
print(f"Saved optical flow video to {flow_video_path}")

# option 2: comparison of frame vs flow
comparison_frames = []
# Loop over the CPU tensors
for img1, flow_img in zip(img1_batch_viz, flow_imgs):
    # img1 is float [0,1], flow_img is byte [0,255]. Need to match types.
    # Convert img1 to byte [0,255] for concatenation
    img1 = (img1 * 255).byte()
    
    # concatenate horizontally [C, H, W] + [C, H, W] -> [C, H, 2*W]
    combined = torch.cat([img1, flow_img], dim=2)
    comparison_frames.append(combined)

comparison_video = torch.stack(comparison_frames, dim=0)  # [T, C, H, 2*W]
comparison_path = os.path.join(output_dir, f"{name}_comparison.mp4")
torchvision.io.write_video(comparison_path, comparison_video.permute(0, 2, 3, 1), fps=30)
print(f"Saved comparison video to {comparison_path}")