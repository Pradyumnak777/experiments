import os
import sys
import pickle
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from pathlib import Path
from collections import OrderedDict
from matplotlib import cm

#setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COTRACKER_ROOT = Path(__file__).resolve().parent / "co-tracker"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.utils.visualizer import Visualizer
from point_sampling.model_finetune import DINOv2_LoRA, get_robust_mask
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#hardcoding paths for the comparison
video_coach_path = "UCF_Rep/val/v_BodyWeightSquats_g21_c04.mp4"
video_student_path = "UCF_Rep/val/v_BodyWeightSquats_g22_c03.mp4"

if not os.path.isfile(video_coach_path) or not os.path.isfile(video_student_path):
    raise FileNotFoundError("one or both videos not found")

coach_name = Path(video_coach_path).stem
student_name = Path(video_student_path).stem
out_dir_name = f"compare_{coach_name}_vs_{student_name}"
save_dir = os.path.join("point_track/saved_videos", out_dir_name)
os.makedirs(save_dir, exist_ok=True)

#load raw video data
frames_coach = iio.imread(video_coach_path, plugin="FFMPEG")
meta_coach = iio.immeta(video_coach_path, plugin="FFMPEG")
fps_coach = float(meta_coach.get("fps", 30))

frames_student = iio.imread(video_student_path, plugin="FFMPEG")
meta_student = iio.immeta(video_student_path, plugin="FFMPEG")
fps_student = float(meta_student.get("fps", 30))

raft_transform = v2.Compose([
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    v2.Resize(size=(520, 960)),
])

def build_three_frame_chunk(frames_np, start_frame=20, stride=2, num_frames=3, target_size=(224, 224)):
    idxs = [start_frame + i * stride for i in range(num_frames)]
    if idxs[-1] >= len(frames_np):
        raise ValueError("not enough frames for sampling")

    sampled = torch.from_numpy(frames_np[idxs]).permute(0, 3, 1, 2).float() / 255.0
    sampled = F.interpolate(sampled, size=target_size, mode="bilinear", align_corners=False)
    raw_frames = sampled.permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().cpu().numpy()

    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    pixel_frames = (sampled.to(device) - norm_mean) / norm_std

    return raw_frames, pixel_frames

def farthest_point_sampling_2d(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    if points.shape[0] <= num_samples:
        return points

    points_float = points.float()
    selected_indices = torch.empty(num_samples, dtype=torch.long, device=points.device)
    center = points_float.mean(dim=0, keepdim=True)
    farthest_index = torch.argmax(torch.sum((points_float - center) ** 2, dim=1))
    min_distances = torch.full((points.shape[0],), float("inf"), device=points.device)

    for sample_index in range(num_samples):
        selected_indices[sample_index] = farthest_index
        selected_point = points_float[farthest_index : farthest_index + 1]
        distances = torch.sum((points_float - selected_point) ** 2, dim=1)
        min_distances = torch.minimum(min_distances, distances)
        farthest_index = torch.argmax(min_distances)

    return points[selected_indices]

#prepare full video tensors for cotracker later
video_tensor_coach = torch.tensor(frames_coach).permute(0, 3, 1, 2)[None].float().to(device)
video_tensor_student = torch.tensor(frames_student).permute(0, 3, 1, 2)[None].float().to(device)

print("loading custom dino and raft models...")
model = DINOv2_LoRA().to(device)
model_path = "test_models/physics_guide_lora_dino_epoch_9.pth" 
state_dict = torch.load(model_path, map_location=device, weights_only=True)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

raft_model = raft_large(pretrained=True, progress=False).to(device).eval()

start_f_coach = 20
start_f_student = 20 
mask_frame_idx = 1
query_frame_coach = start_f_coach + (mask_frame_idx * 2)
query_frame_student = start_f_student + (mask_frame_idx * 2)

raw_coach, px_coach = build_three_frame_chunk(frames_coach, start_frame=start_f_coach)
raw_student, px_student = build_three_frame_chunk(frames_student, start_frame=start_f_student)

#i only need raft flow for the coach to find the anchor points
print("generating optical flow for coach template...")
flow_list = []
for i in range(2):
    img1 = raft_transform(torch.from_numpy(raw_coach[i]).permute(2, 0, 1)).to(device).unsqueeze(0)
    img2 = raft_transform(torch.from_numpy(raw_coach[i + 1]).permute(2, 0, 1)).to(device).unsqueeze(0)
    with torch.no_grad():
        list_of_flows = raft_model(img1, img2)
        flow_res = F.interpolate(list_of_flows[-1], size=(224, 224), mode="bilinear", align_corners=False)
        flow_list.append(flow_res.squeeze(0))

flow_list.append(flow_list[-1].clone())
flow_tensor = torch.stack(flow_list)

print("running dino inference for semantic mapping...")
with torch.no_grad():
    #pass coach
    out_coach = model(px_coach.unsqueeze(0))
    pred_mask_coach = out_coach["pred_mask"]
    feat_coach = out_coach["patch_features"][:, mask_frame_idx]  # (1, 768, 16, 16)
    
    #pass student
    out_student = model(px_student.unsqueeze(0))
    feat_student = out_student["patch_features"][:, mask_frame_idx]  # (1, 768, 16, 16)

pred_mask_224 = F.interpolate(pred_mask_coach[0], size=(224, 224), mode="bilinear", align_corners=False)
mask_map = pred_mask_224[mask_frame_idx, 0]

mask_threshold = 0.3
candidate_coords = torch.nonzero(mask_map > mask_threshold, as_tuple=False)

max_query_points = 100
if candidate_coords.shape[0] > max_query_points:
    coach_224_coords = farthest_point_sampling_2d(candidate_coords, max_query_points)
else:
    coach_224_coords = candidate_coords

#upsample the 16x16 features to 224x224 so i can map pixel to pixel
feat_coach_224 = F.interpolate(feat_coach, size=(224, 224), mode="bilinear", align_corners=False)
feat_student_224 = F.interpolate(feat_student, size=(224, 224), mode="bilinear", align_corners=False)

feat_coach_224 = F.normalize(feat_coach_224, p=2, dim=1)
feat_student_224 = F.normalize(feat_student_224, p=2, dim=1)

print("projecting coach points to student video...")
#extract the specific 768-d vectors for the sampled coach points
y_c = coach_224_coords[:, 0]
x_c = coach_224_coords[:, 1]
#shape becomes [768, num_points]
sampled_coach_feats = feat_coach_224[0, :, y_c, x_c] 

#flatten student feature map to [768, 224*224] for matrix multiplication
flat_student_feats = feat_student_224[0].view(768, -1)

#compute cosine similarity for all pixels
sim_maps = torch.matmul(sampled_coach_feats.T, flat_student_feats) # [num_points, 50176]

#find the highest similarity pixel in the student frame for each coach point
best_match_indices = torch.argmax(sim_maps, dim=1)

#convert flat indices back to y,x coordinates
y_s = best_match_indices // 224
x_s = best_match_indices % 224

#free up gpu memory before loading cotracker
del raft_model
del model
del flow_tensor
del out_coach
del out_student
del feat_coach_224
del feat_student_224
torch.cuda.empty_cache()

print("loading cotracker for joint tracking...")
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

_, _, _, h_coach, w_coach = video_tensor_coach.shape
_, _, _, h_student, w_student = video_tensor_student.shape

#scale coordinates from 224 to original resolutions
y_coach_orig = y_c.float() * ((h_coach - 1) / (224 - 1))
x_coach_orig = x_c.float() * ((w_coach - 1) / (224 - 1))
t_coach = torch.full_like(x_coach_orig, float(query_frame_coach))
queries_coach = torch.stack([t_coach, x_coach_orig, y_coach_orig], dim=1).unsqueeze(0).to(device)

y_student_orig = y_s.float() * ((h_student - 1) / (224 - 1))
x_student_orig = x_s.float() * ((w_student - 1) / (224 - 1))
t_student = torch.full_like(x_student_orig, float(query_frame_student))
queries_student = torch.stack([t_student, x_student_orig, y_student_orig], dim=1).unsqueeze(0).to(device)

print(f"tracking {queries_coach.shape[1]} semantically linked points in both videos...")

#run tracker independently but they are locked semantically
pred_tracks_coach, pred_vis_coach = cotracker(video_tensor_coach, queries=queries_coach)
pred_tracks_student, pred_vis_student = cotracker(video_tensor_student, queries=queries_student)

#save data
save_data = {
    "coach": {
        "video_path": video_coach_path,
        "query_frame": int(query_frame_coach),
        "tracks": pred_tracks_coach.detach().cpu().numpy(),
        "visibility": pred_vis_coach.detach().cpu().numpy(),
        "fps": fps_coach
    },
    "student": {
        "video_path": video_student_path,
        "query_frame": int(query_frame_student),
        "tracks": pred_tracks_student.detach().cpu().numpy(),
        "visibility": pred_vis_student.detach().cpu().numpy(),
        "fps": fps_student
    }
}

tracks_path = os.path.join(save_dir, "paired_trajectories.pkl")
with open(tracks_path, "wb") as f:
    pickle.dump(save_data, f)
print(f"saved paired trajectories to {tracks_path}")

#visualize both
vis_coach = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, fps=fps_coach)
vis_coach.visualize(video_tensor_coach, pred_tracks_coach, pred_vis_coach, filename=f"{coach_name}_tracks")

vis_student = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, fps=fps_student)
vis_student.visualize(video_tensor_student, pred_tracks_student, pred_vis_student, filename=f"{student_name}_tracks")