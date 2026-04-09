import os
import sys
import pickle
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from pathlib import Path
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

#setup paths to match the project structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

#import the custom finetuned model
from point_sampling.model_finetune import DINOv2_LoRA

vid_1_info = "point_track/saved_videos/v_BodyWeightSquats_g21_c04/v_BodyWeightSquats_g21_c04_trajectories_frame_22.pkl"
vid_2_info = "point_track/saved_videos/v_BodyWeightSquats_g22_c03/v_BodyWeightSquats_g22_c03_trajectories_frame_22.pkl"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#helper function to load pickle data
def load_track_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

#helper to pull out a single frame and prep it for dino inference
def get_dino_input(video_path, frame_idx):
    frames = iio.imread(video_path, plugin="FFMPEG")
    frame_np = frames[frame_idx]
    
    #convert to tensor and normalize exactly how i did in the pipe
    frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
    frame_t = F.interpolate(frame_t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    normalized_frame = (frame_t.to(device) - norm_mean) / norm_std
    
    #the custom forward pass expects (b, t, c, h, w) so i add a dummy time dimension
    final_input = normalized_frame.unsqueeze(1)
    
    return final_input, frames.shape[2], frames.shape[1]

#load up the saved trajectory data for both videos
data_1 = load_track_data(vid_1_info)
data_2 = load_track_data(vid_2_info)

#grab the starting coordinates from the exact query frame
#tracks shape is usually (1, t, n, 2)
points_1 = torch.tensor(data_1["tracks"][0, data_1["query_frame"]]).to(device)
points_2 = torch.tensor(data_2["tracks"][0, data_2["query_frame"]]).to(device)

print("loading finetuned dino model...")
model_path = "test_models/physics_guide_lora_dino_epoch_9.pth" 
model = DINOv2_LoRA().to(device)

state_dict = torch.load(model_path, map_location=device, weights_only=True)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

#extract the query frames and original dimensions
img_1, w1, h1 = get_dino_input(data_1["video_path"], data_1["query_frame"])
img_2, w2, h2 = get_dino_input(data_2["video_path"], data_2["query_frame"])

print("extracting feature maps...")
with torch.no_grad():
    #pass through the custom forward method to get the properly shaped dictionary
    out_1 = model(img_1)
    out_2 = model(img_2)
    
    #grab patch features and drop the dummy time dimension
    #shape goes from (1, 1, 768, 16, 16) to (1, 768, 16, 16)
    feat_1 = out_1["patch_features"].squeeze(1)
    feat_2 = out_2["patch_features"].squeeze(1)
    
    #l2 normalize the feature maps for cosine similarity later
    feat_1 = F.normalize(feat_1, p=2, dim=1)
    feat_2 = F.normalize(feat_2, p=2, dim=1)

#normalize the tracked coordinates to the [-1, 1] range needed for grid_sample
#cotracker outputs standard pixel coords so they are mapped based on original video resolution
norm_x_1 = (points_1[:, 0] / (w1 - 1)) * 2 - 1
norm_y_1 = (points_1[:, 1] / (h1 - 1)) * 2 - 1
grid_1 = torch.stack([norm_x_1, norm_y_1], dim=-1).unsqueeze(0).unsqueeze(0) 

norm_x_2 = (points_2[:, 0] / (w2 - 1)) * 2 - 1
norm_y_2 = (points_2[:, 1] / (h2 - 1)) * 2 - 1
grid_2 = torch.stack([norm_x_2, norm_y_2], dim=-1).unsqueeze(0).unsqueeze(0) 

print("sampling semantic features for all tracked points...")
#sample the exact feature vectors at the point locations
#output shape will be (1, c, 1, num_points)
sampled_feats_1 = F.grid_sample(feat_1, grid_1, align_corners=False).squeeze().T 
sampled_feats_2 = F.grid_sample(feat_2, grid_2, align_corners=False).squeeze().T 

#calculate the dense cosine similarity matrix between all points in vid 1 and vid 2
#matrix shape will be (num_points_1, num_points_2)
similarity_matrix = torch.matmul(sampled_feats_1, sampled_feats_2.T)

#convert to a cost matrix for the hungarian matching algorithm
#subtract from 1 because the algorithm minimizes cost but i want maximum similarity
cost_matrix = 1.0 - similarity_matrix.cpu().numpy()

#set k to the number of best matches i want to keep
K = 20

print("running bipartite matching...")
#find the optimal 1-to-1 semantic correspondence for all points first
row_ind, col_ind = linear_sum_assignment(cost_matrix)

#collect all matches with their scores to sort them
all_matches = []
for r, c in zip(row_ind, col_ind):
    sim_score = similarity_matrix[r, c].item()
    all_matches.append((r, c, sim_score))

#sort by similarity score in descending order
all_matches.sort(key=lambda x: x[2], reverse=True)

#slice the top k matches
top_k_matches = all_matches[:K]

print(f"\ntop {K} semantic point matches:")
mapping_dict = {}
for i, (idx_1, idx_2, score) in enumerate(top_k_matches):
    mapping_dict[int(idx_1)] = int(idx_2)
    if i < 5: #just print the first 5 to check
        print(f"vid 1 point {idx_1} maps to vid 2 point {idx_2} (cosine sim: {score:.4f})")

#save out the filtered mapping dictionary
out_path = Path("point_track/saved_paths/test1.pkl")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "wb") as f:
    pickle.dump(mapping_dict, f)
print(f"\nsaved top {len(mapping_dict)} correspondence map to {out_path}")