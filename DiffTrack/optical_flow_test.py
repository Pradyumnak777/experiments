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


device = "cuda" if torch.cuda.is_available() else "cpu"
frames = read_frames_as_video("videos/benchpress")
img1_batch = frames[:-1]
img2_batch = frames[1:]

img1_batch = preprocess(img1_batch).to(device)
img2_batch = preprocess(img2_batch).to(device)

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

predicted_flows = list_of_flows[-1]

flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

# optical flow save as video
output_dir = "output_optical_flow"
os.makedirs(output_dir, exist_ok=True)

# option 1: flow visualization
flow_video_path = os.path.join(output_dir, "optical_flow.mp4")
torchvision.io.write_video(flow_video_path, flow_imgs.permute(0, 2, 3, 1).cpu(), fps=30)
print(f"Saved optical flow video to {flow_video_path}")

# option 2: comparison of frame vs flow
comparison_frames = []
for img1, flow_img in zip(img1_batch, flow_imgs):
    # concatenate horizontally [C, H, W] + [C, H, W] -> [C, H, 2*W]
    combined = torch.cat([img1.cpu(), flow_img.cpu()], dim=2)
    comparison_frames.append(combined)

comparison_video = torch.stack(comparison_frames, dim=0)  # [T, C, H, 2*W]
comparison_path = os.path.join(output_dir, "comparison.mp4")
torchvision.io.write_video(comparison_path, comparison_video.permute(0, 2, 3, 1), fps=30)
print(f"Saved comparison video to {comparison_path}")
