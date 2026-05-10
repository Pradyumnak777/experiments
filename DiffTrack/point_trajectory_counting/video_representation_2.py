import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import imageio.v3 as iio
import imageio
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

#this tiny mlp replaces pca
#it compresses the 200 coordinates into a 64-dimensional rhythm space
class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TrajectoryEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        
    def forward(self, x):
        return self.encoder(x)

def resample_traj(traj, speed):
    #mathematically warp the trajectory to create fake speed labels
    t_orig = len(traj)
    t_new = int(t_orig / speed)
    indices = np.linspace(0, t_orig - 1, t_new)
    resampled = np.zeros((t_new, traj.shape[1], 2))
    for p in range(traj.shape[1]):
        for c in range(2):
            resampled[:, p, c] = np.interp(indices, np.arange(t_orig), traj[:, p, c])
    return resampled

def simper_loss(features, speeds, temperature=0.1):
    #generalized contrastive loss forcing rhythm sorting
    sim_matrix = torch.matmul(features, features.T) / temperature
    speed_diff = torch.abs(speeds.unsqueeze(0) - speeds.unsqueeze(1))
    soft_targets = torch.exp(-speed_diff) 
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
    
    log_probs = torch.log_softmax(sim_matrix, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss

def extract_rgb_features(vid_path):
    #load a lightweight pre-trained cnn to act as our spatial eyes
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frames = iio.imread(vid_path, plugin="FFMPEG")
    features = []
    
    #process frame by frame to extract spatial context
    with torch.no_grad():
        for frame in frames:
            tensor_img = transform(frame).unsqueeze(0)
            feat = model(tensor_img).squeeze()
            features.append(feat.numpy())
            
    return np.array(features), frames

def extract_simper_trajectory_features(traj_file, t_frames):
    #load the tracking data
    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)
        
    tracks = trajectories['tracks'][0]
    t, n, _ = tracks.shape
    
    #convert raw pixel coordinates to scale-invariant unit velocities
    vel = np.diff(tracks, axis=0, prepend=tracks[:1])
    mags = np.linalg.norm(vel, axis=-1, keepdims=True)
    unit_vel = vel / (mags + 1e-6)
    
    #initialize the miniature rhythm encoder and optimizer
    model = TrajectoryEncoder(input_dim=n*2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    speed_range = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5]
    
    print("training video-specific simper encoder on 100 points...")
    model.train()
    
    #overfit to this specific video for 50 epochs to act as a noise filter
    for epoch in range(50):
        epoch_features = []
        epoch_speeds = []
        
        for s in speed_range:
            warped = resample_traj(unit_vel, s)
            flat_warped = torch.tensor(warped.reshape(warped.shape[0], -1), dtype=torch.float32)
            feat = model(flat_warped).mean(dim=0)
            epoch_features.append(feat)
            epoch_speeds.append(s)
            
        features_stack = torch.stack(epoch_features)
        speeds_stack = torch.tensor(epoch_speeds, dtype=torch.float32)
        
        loss = simper_loss(features_stack, speeds_stack)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"simper optimization complete. final loss: {loss.item():.4f}")
    
    #extract the cleaned features for the original 1.0x speed video
    model.eval()
    with torch.no_grad():
        flat_input = torch.tensor(unit_vel.reshape(t, -1), dtype=torch.float32)
        motion_features = model(flat_input).numpy()
    
    return motion_features[:t_frames]

def build_fused_tssm(vid_path, traj_file):
    #grab the visual features from the pre-trained resnet
    print("extracting rgb features...")
    rgb_feats, frames = extract_rgb_features(vid_path)
    t = rgb_feats.shape[0]
    
    #grab the motion features by training our simper encoder on the fly
    print("extracting trajectory features...")
    traj_feats = extract_simper_trajectory_features(traj_file, t)
    
    #normalize both feature sets independently for equal voting power
    rgb_norm = rgb_feats / (np.linalg.norm(rgb_feats, axis=1, keepdims=True) + 1e-6)
    traj_norm = traj_feats / (np.linalg.norm(traj_feats, axis=1, keepdims=True) + 1e-6)
    
    #early fusion concatenate spatial and temporal embeddings
    fused_feats = np.concatenate([rgb_norm, traj_norm], axis=1)
    fused_norm = fused_feats / (np.linalg.norm(fused_feats, axis=1, keepdims=True) + 1e-6)
    
    #build the master checkerboard matrix
    tssm = np.dot(fused_norm, fused_norm.T)
    
    return tssm, frames

if __name__ == "__main__":
    vid_path = "UCF_Rep/val/v_PlayingViolin_g21_c01.mp4"
    traj_file = "point_track/saved_videos/v_PlayingViolin_g21_c01/v_PlayingViolin_g21_c01_trajectories_frame_22.pkl"
    
    vid_name = os.path.basename(vid_path).replace(".mp4", "")
    output_vid_path = f"point_trajectory_counting/{vid_name}_fused_simper_tssm_viz.mp4"
    os.makedirs("point_trajectory_counting", exist_ok=True)
    
    #generate the fused matrix data using our on-the-fly trained encoder
    tssm, frames = build_fused_tssm(vid_path, traj_file)
    t = len(frames)
    
    print(f"generating video: {output_vid_path}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    with imageio.get_writer(output_vid_path, fps=30, codec='libx264', quality=8) as writer:
        for cur_t in range(t):
            ax1.clear(); ax2.clear()
            
            #panel for the original video frame
            ax1.imshow(frames[cur_t])
            ax1.set_title("input video")
            ax1.axis('off')
            
            #panel for the fused rgb + learned trajectory tssm
            ax2.imshow(tssm, cmap='magma', origin='lower')
            ax2.axvline(cur_t, color='cyan', alpha=0.5)
            ax2.axhline(cur_t, color='cyan', alpha=0.5)
            ax2.set_title("fused rgb + simper trajectory tssm")
            
            canvas.draw()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            frame_to_save = buf.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]
            writer.append_data(frame_to_save)
            
            if cur_t % 50 == 0:
                print(f"processed frame {cur_t}/{t}")

    plt.close(fig)