import os
import pickle
import torch
import imageio.v3 as iio
import sys
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
from matplotlib import cm

GPU_ID = 1  
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COTRACKER_ROOT = Path(__file__).resolve().parent / "co-tracker"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.utils.visualizer import Visualizer
from point_sampling.model_finetune import DINOv2_LoRA, get_robust_mask
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import v2




video_path = "UCF_Rep/val/v_TrampolineJumping_g25_c03.mp4"
if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

frames = iio.imread(video_path, plugin="FFMPEG")
meta = iio.immeta(video_path, plugin="FFMPEG")
source_fps = float(meta.get("fps", 30))

input_name = Path(video_path).stem
out_name = f"{input_name}_cotracker"
save_dir = os.path.join("point_track/saved_videos", input_name)
os.makedirs(save_dir, exist_ok=True)


device = 'cuda'


raft_transform = v2.Compose([
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    v2.Resize(size=(520, 960)),
])


def build_three_frame_chunk_from_loaded_frames(
    frames_np,
    start_frame=40,
    stride=2,
    num_frames=3,
    target_size=(224, 224),
    device="cuda",
):
    idxs = [start_frame + i * stride for i in range(num_frames)]
    if idxs[-1] >= len(frames_np):
        raise ValueError(
            f"Not enough frames for sampling: need index {idxs[-1]}, but only {len(frames_np)} frames available"
        )

    sampled = torch.from_numpy(frames_np[idxs]).permute(0, 3, 1, 2).float() / 255.0
    sampled = F.interpolate(sampled, size=target_size, mode="bilinear", align_corners=False)

    raw_frames = (
        sampled.permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().cpu().numpy()
    )

    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    pixel_frames = (sampled.to(device) - norm_mean) / norm_std

    return raw_frames, pixel_frames


# def farthest_point_sampling_2d(points: torch.Tensor, num_samples: int) -> torch.Tensor:
#     if points.shape[0] <= num_samples:
#         return points

#     points_float = points.float()
#     selected_indices = torch.empty(num_samples, dtype=torch.long, device=points.device)

#     center = points_float.mean(dim=0, keepdim=True)
#     farthest_index = torch.argmax(torch.sum((points_float - center) ** 2, dim=1))
#     min_distances = torch.full((points.shape[0],), float("inf"), device=points.device)

#     for sample_index in range(num_samples):
#         selected_indices[sample_index] = farthest_index
#         selected_point = points_float[farthest_index : farthest_index + 1]
#         distances = torch.sum((points_float - selected_point) ** 2, dim=1)
#         min_distances = torch.minimum(min_distances, distances)
#         farthest_index = torch.argmax(min_distances)

#     return points[selected_indices]

def hybrid_point_sampling_2d(
    points: torch.Tensor, 
    mask_values: torch.Tensor, 
    num_samples: int, 
    top_k_ratio: float = 0.5
) -> torch.Tensor:
    """
    Samples points using a mix of highest mask intensity and Farthest Point Sampling.
    """
    if points.shape[0] <= num_samples:
        return points

    # Determine how many points go to Top-K vs FPS
    num_top_k = int(num_samples * top_k_ratio)
    num_fps = num_samples - num_top_k

    # 1. Select Top-K highest intensity points
    _, top_indices = torch.topk(mask_values, num_top_k)
    selected_indices = top_indices.tolist()

    if num_fps <= 0:
        return points[selected_indices]

    # 2. Run FPS for the remainder
    points_float = points.float()
    min_distances = torch.full((points.shape[0],), float("inf"), device=points.device)

    # CRITICAL: Initialize distances to the already selected Top-K points
    # This forces the FPS to spread *away* from the dense clusters
    selected_points_tensor = points_float[selected_indices]
    for pt in selected_points_tensor:
        distances = torch.sum((points_float - pt) ** 2, dim=1)
        min_distances = torch.minimum(min_distances, distances)

    # 3. Standard FPS loop for the remaining points
    for _ in range(num_fps):
        farthest_index = torch.argmax(min_distances).item()
        selected_indices.append(farthest_index)

        selected_point = points_float[farthest_index : farthest_index + 1]
        distances = torch.sum((points_float - selected_point) ** 2, dim=1)
        min_distances = torch.minimum(min_distances, distances)

    return points[selected_indices]


video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

#using custom point sampler instead of normal grid sampling
model_path = "test_models/physics_guide_lora_dino_epoch_9.pth" 
print("loading models(DINO + RAFT) onto gpu...")
model = DINOv2_LoRA().to(device)

state_dict = torch.load(model_path, map_location=device, weights_only=True)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

raft_model = raft_large(pretrained=True, progress=False).to(device).eval()

'''
FRAME POSITION MATTERS! IS THERE A WAY TO AUTOMATE THIS???
'''

start_f = 20 #pick a frame in the middle

raw_frames, pixel_frames = build_three_frame_chunk_from_loaded_frames(
    frames,
    start_frame=start_f,
    stride=2,
    num_frames=3,
    target_size=(224, 224),
    device=device,
)

print("generating optical flow...")
flow_list = []
for i in range(2):
    img1 = raft_transform(torch.from_numpy(raw_frames[i]).permute(2, 0, 1)).to(device).unsqueeze(0)
    img2 = raft_transform(torch.from_numpy(raw_frames[i + 1]).permute(2, 0, 1)).to(device).unsqueeze(0)

    with torch.no_grad():
        list_of_flows = raft_model(img1, img2)
        flow_res = F.interpolate(list_of_flows[-1], size=(224, 224), mode="bilinear", align_corners=False)
        flow_list.append(flow_res.squeeze(0))

flow_list.append(flow_list[-1].clone())
flow_tensor = torch.stack(flow_list)

print("running DINO finetuned mask inference...")
input_pixels = pixel_frames.unsqueeze(0)
with torch.no_grad():
    outputs = model(input_pixels)
    pred_mask = outputs["pred_mask"]
    teacher_mask = get_robust_mask(flow_tensor.unsqueeze(0))

pred_mask_224 = F.interpolate(
    pred_mask[0], size=(224, 224), mode="bilinear", align_corners=False
)
print(f"pred_mask shape: {tuple(pred_mask.shape)}")
print(f"pred_mask_224 shape: {tuple(pred_mask_224.shape)}")
print(f"teacher_mask shape: {tuple(teacher_mask.shape)}")

'''
now inspecting the mask, and setting a threshold based on median score/intensity.
Select all points over threshold as query and pass into co-tracker..
'''

mask_frame_idx = 1
sampled_frame_indices = [start_f + i * 2 for i in range(3)]
query_frame = sampled_frame_indices[mask_frame_idx]

teacher_mask_map = teacher_mask[0, mask_frame_idx, 0].detach().cpu().numpy().astype("float32")
teacher_frame = raw_frames[mask_frame_idx].astype("float32")
teacher_mask_color = (cm.get_cmap("jet")(teacher_mask_map)[..., :3] * 255.0).astype("float32")
teacher_alpha = (0.5 * teacher_mask_map)[..., None]
teacher_overlay = (teacher_frame * (1.0 - teacher_alpha) + teacher_mask_color * teacher_alpha).clip(0, 255).astype("uint8")

teacher_mask_name = f"{input_name}_teacher_mask_overlay_frame_{query_frame}.png"
teacher_mask_path = os.path.join(save_dir, teacher_mask_name)
iio.imwrite(teacher_mask_path, teacher_overlay)
print(f"Saved teacher mask overlay to {teacher_mask_path}")

mask_map = pred_mask_224[mask_frame_idx, 0]
mask_threshold = 0.2
candidate_coords = torch.nonzero(mask_map > mask_threshold, as_tuple=False)

max_query_points = 50
if candidate_coords.shape[0] > max_query_points:
    #Extract the actual mask values for those specific coordinates
    # We use candidate_coords[:, 0] for Y and candidate_coords[:, 1] for X
    candidate_intensities = mask_map[candidate_coords[:, 0], candidate_coords[:, 1]] #getting finetuned dino scores for these points..
    
    # 3. Pass both to the new hybrid sampler (e.g., reserving 40% of points for the densest areas)
    candidate_coords = hybrid_point_sampling_2d(
        candidate_coords, 
        candidate_intensities, 
        max_query_points, 
        top_k_ratio=0.05 # Adjust this up or down depending on how much clustering you want
    )

pred_mask_map = pred_mask_224[mask_frame_idx, 0].detach().cpu().numpy().astype("float32")
pred_frame = raw_frames[mask_frame_idx].astype("float32")
pred_mask_color = (cm.get_cmap("jet")(pred_mask_map)[..., :3] * 255.0).astype("float32")
pred_alpha = pred_mask_map[..., None]
pred_overlay = (pred_frame * (1.0 - pred_alpha) + pred_mask_color * pred_alpha).clip(0, 255).astype("uint8")

pred_mask_name = f"{input_name}_pred_mask_overlay_frame_{query_frame}.png"
pred_mask_path = os.path.join(save_dir, pred_mask_name)
iio.imwrite(pred_mask_path, pred_overlay)
print(f"Saved predicted mask overlay to {pred_mask_path}")

del raft_model
del model
del flow_tensor
del outputs
del pred_mask
del teacher_mask
if device == "cuda":
    torch.cuda.empty_cache()

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

if candidate_coords.shape[0] == 0:
    print("No mask-based query points found.. falling back to grid sampling.")
    pred_tracks, pred_visibility = cotracker(video, grid_size=10)
else:
    _, _, _, h_orig, w_orig = video.shape
    y = candidate_coords[:, 0].float() * ((h_orig - 1) / (224 - 1))
    x = candidate_coords[:, 1].float() * ((w_orig - 1) / (224 - 1))
    t = torch.full_like(x, float(query_frame))

    queries = torch.stack([t, x, y], dim=1).unsqueeze(0).to(device)
    print(
        f"Using {queries.shape[1]} custom queries from mask frame {query_frame} (threshold={mask_threshold:.4f})"
    )
    pred_tracks, pred_visibility = cotracker(video, queries=queries)

tracks_name = f"{input_name}_trajectories_frame_{query_frame}.pkl"
tracks_path = os.path.join(save_dir, tracks_name)
with open(tracks_path, "wb") as file_handle:
    pickle.dump(
        {
            "video_path": video_path,
            "query_frame": int(query_frame),
            "mask_threshold": float(mask_threshold),
            "tracks": pred_tracks.detach().cpu().numpy(),
            "visibility": pred_visibility.detach().cpu().numpy(),
            "fps": float(source_fps),
        },
        file_handle,
    )
print(f"Saved trajectories to {tracks_path}")

vis = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, fps=source_fps)
vis.visualize(video, pred_tracks, pred_visibility, filename=out_name)
