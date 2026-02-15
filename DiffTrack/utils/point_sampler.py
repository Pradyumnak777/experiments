import os
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image

dir = "optical_flow_tensors"
flow_tensors_list = []

for filename in os.listdir(dir):
    filepath = os.path.join(dir, filename)
    # extract name before '_flow'
    if '_flow' in filename:
        name = filename.split('_flow')[0]
    else:
        name = os.path.splitext(filename)[0]
    with open(filepath, "rb") as f:
        flow_array = pickle.load(f)
        flow_tensor = torch.from_numpy(flow_array).float() #conversion to tensor
        flow_tensors_list.append((name, flow_tensor))

#now do analysis..
'''
structure of each flow tensor: (N, 2, H, W)
(u,v) represent 2, which is x-displacement and y-displacement
u>0: moved right, u<0: moved left
v>0: moved down, v<0: moved up
'''

'''
IDEA 1: optical flow!

sample a grid. and look at motion within each grid? and sleect the "best" one from each grid..
'''


# points_to_sample = 10
# grid_size = 40 #each grid is 40x40...

# for name, flow_tensor in flow_tensors_list:
#     flow_mag = torch.norm(flow_tensor, dim=1) 
#     heatmap = torch.std(flow_mag, dim=0) # (H, W)
#     H, W = heatmap.shape
    
#     global_mean = torch.mean(heatmap).item()
#     grid_candidates = []
#     points_to_track = []
    
#     for y in range(0, H, grid_size):
#         for x in range(0, W, grid_size):
            
#             #creating grid boxes
#             y_end = min(y + grid_size, H)
#             x_end = min(x + grid_size, W)
            
#             # Slice the heatmap for this specific box
#             box_values = heatmap[y:y_end, x:x_end]
            
#             # CHECK: Is this box active?
#             # We look at the MAX variance inside this box.
#             peak_score = torch.max(box_values).item()
            
#             # If the peak activity here is less than 1.5x the global average, it's just noise.
#             if peak_score < (global_mean * 1.5): 
#                 continue
            
#             #pick best index in this box
#             max_idx = torch.argmax(box_values)
#             local_y, local_x = np.unravel_index(max_idx.cpu().numpy(), box_values.shape)
            
#             # Convert to global coordinates
#             global_y = y + local_y
#             global_x = x + local_x
            
#             # Store candidate: (Score, x, y)
#             grid_candidates.append((peak_score, global_x, global_y))
            
#     grid_candidates.sort(key=lambda x: x[0], reverse=True)
#     final_points = grid_candidates[:points_to_sample]
    
#     points_to_track = []
#     for score, x, y in final_points:
#         points_to_track.append([0, x, y])

#     # Convert to Tensor
#     points_tensor = torch.tensor(points_to_track).float()
    
#     data_to_save = {
#         "points": points_tensor.cpu().numpy(),
#         "orig_h": H,
#         "orig_w": W
#     }
    
#     os.makedirs("points_to_sample", exist_ok=True)
#     with open(f"points_to_sample/{name}_points.pkl", "wb") as f:
#         pickle.dump(data_to_save, f)
        
#     print(f"[{name}] Saved {len(points_to_track)} points. (found {len(grid_candidates)} active grid cells)")
    

'''
IDEA 2- uusing dinv2..
'''
print("Loading DINOv2 model...")
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
dino_model.eval()
    
VIDEO_DIR = "videos" 
points_to_sample = 5
grid_size = 40

def get_semantic_mask(img_path, threshold_percentile=60):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    img_tensor = transform(img).unsqueeze(0).cuda()
    
    # get the features
    with torch.no_grad():
        features_dict = dino_model.forward_features(img_tensor)
        features = features_dict['x_norm_patchtokens'] # feats of 14x14 patches in this image(?)
        
    # doing pca to find the "main object"
    features = features.cpu().numpy()[0] 
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features) # so 3 dims
    
    # turn the 1st principal component back into a 2d map/pic
    patch_h, patch_w = 518 // 14, 518 // 14
    foreground_map = pca_features[:, 0].reshape(patch_h, patch_w) 
    
    # normalize
    foreground_map = (foreground_map - foreground_map.min()) / (foreground_map.max() - foreground_map.min())
    
    # resize og dimensions
    foreground_map = torch.tensor(foreground_map).unsqueeze(0).unsqueeze(0)
    foreground_mask_hires = F.interpolate(foreground_map, size=(h, w), mode='bilinear').squeeze().numpy()
    
    # binary mask
    threshold = np.percentile(foreground_mask_hires, threshold_percentile)
    
    #corner background flip.. (can remove?)
    if foreground_mask_hires[0,0] > threshold:
        foreground_mask_hires = 1 - foreground_mask_hires
        threshold = np.percentile(foreground_mask_hires, threshold_percentile)

    binary_mask = foreground_mask_hires > threshold
    return binary_mask

for name, _ in flow_tensors_list:
    
    
    video_path = os.path.join(VIDEO_DIR, name)
    frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
    
    if len(frames) == 0:
        print(f"[{name}] skipping.. no frames found in {video_path}")
        continue
    
    #using middle frame
    middle_frame_path = os.path.join(video_path, frames[len(frames)//2])
    
    try:
        # returns true/1 ehre the "actor" is.
        mask = get_semantic_mask(middle_frame_path, threshold_percentile=70) 
        H, W = mask.shape
    except Exception as e:
        print(f"[{name}] dino failed: {e}")
        continue

    grid_candidates = []
    
    #now basically doing the same grid logic as idea 1, but checking the dino mask instead of flow
    for y in range(0, H, grid_size):
        for x in range(0, W, grid_size):
            
            # creating grid boxes
            y_end = min(y + grid_size, H)
            x_end = min(x + grid_size, W)
            
            # slice the mask for this box
            box_mask = mask[y:y_end, x:x_end]
            
            # if over 50% "Trues" in this grid box, then pick a point from here
            if np.mean(box_mask) > 0.5:
                
                #picking center point
                center_y = y + (grid_size // 2)
                center_x = x + (grid_size // 2)
                
                # [t, x, y]
                grid_candidates.append([0, center_x, center_y])
    
    # # if we have too many points, just pick evenly spaced ones to keep coverage
    # if len(grid_candidates) > points_to_sample:
    #     indices = np.linspace(0, len(grid_candidates)-1, points_to_sample, dtype=int)
    #     final_points = [grid_candidates[i] for i in indices]
    # else:
    #     final_points = grid_candidates
    
    #sselecting by DINO confidence?
    grid_candidates.sort(key=lambda x: x[0], reverse=True)
    final_points = grid_candidates[:points_to_sample]
   
   #save
    points_tensor = torch.tensor(final_points).float()
    
    data_to_save = {
        "points": points_tensor.cpu().numpy(),
        "orig_h": H,
        "orig_w": W
    }
    
    os.makedirs("points_to_sample", exist_ok=True)
    with open(f"points_to_sample/{name}_points.pkl", "wb") as f:
        pickle.dump(data_to_save, f)
        
    print(f"[{name}] Saved {len(final_points)} DINO points.")