import os
import torch
import numpy as np
import pickle
import imageio.v3 as iio
import imageio
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, medfilt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def extract_rgb_features(vid_path):
    #load a lightweight pre-trained cnn to act as our spatial eyes
    #resnet18 provides a 512-dimensional embedding per frame
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
    
    #process frame by frame to extract the spatial pose context
    with torch.no_grad():
        for frame in frames:
            tensor_img = transform(frame).unsqueeze(0)
            feat = model(tensor_img).squeeze()
            features.append(feat.numpy())
            
    return np.array(features), frames

def extract_trajectory_features(traj_file, t_frames):
    #load the tracking data from your point tracker
    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)
        
    tracks = trajectories['tracks'][0]
    
    #convert raw pixel coordinates to scale-invariant unit velocities
    #this captures the 'kinematic arc' of the movement
    vel = np.diff(tracks, axis=0, prepend=tracks[:1])
    mags = np.linalg.norm(vel, axis=-1, keepdims=True)
    unit_vel = vel / (mags + 1e-6)
    
    #flatten the point cloud into a single vector per frame
    reshaped_vel = unit_vel.reshape(unit_vel.shape[0], 000-1)
    
    #use pca to extract the shared motion consensus and reduce dimensions
    pca = PCA(n_components=min(64, reshaped_vel.shape[1]))
    motion_features = pca.fit_transform(reshaped_vel)
    
    return motion_features[:t_frames]

def build_fused_tssm(vid_path, traj_file):
    #grab the visual features
    print("extracting rgb features...")
    rgb_feats, frames = extract_rgb_features(vid_path)
    t = rgb_feats.shape[0]
    
    #grab the motion features
    print("extracting trajectory features...")
    traj_feats = extract_trajectory_features(traj_file, t)
    
    #normalize both feature sets independently for equal voting power
    rgb_norm = rgb_feats / (np.linalg.norm(rgb_feats, axis=1, keepdims=True) + 1e-6)
    traj_norm = traj_feats / (np.linalg.norm(traj_feats, axis=1, keepdims=True) + 1e-6)
    
    #early fusion: concatenate spatial and temporal embeddings
    fused_feats = np.concatenate([rgb_norm, traj_norm], axis=1)
    fused_norm = fused_feats / (np.linalg.norm(fused_feats, axis=1, keepdims=True) + 1e-6)
    
    #build the master checkerboard matrix
    tssm = np.dot(fused_norm, fused_norm.T)
    
    return tssm, frames

def get_count_gt(vid_path, labels_dir):
    """Retrieves the ground truth count from the UCF-Rep annotation files."""
    video_name = os.path.splitext(os.path.basename(vid_path))[0]
    video_class = video_name[2:].split("_g", 1)[0]
    
    # Path to your repetition labels
    search_dir = "annotations_ucfrep/repetition_label"
    parts = None
    
    search_file = video_class
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file == f"{search_file}.txt":
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    parts = content.split(video_name)
                break
                
    if parts is not None and len(parts) > 1:
        following_text = parts[1].strip().split('\n\n')[0]
        lines = following_text.splitlines()
        if lines:
            numbers_line = lines[0]
            if numbers_line.strip() == "0":
                return 0, video_name
            count = len(numbers_line.split())
            return count, video_name

    return 0, video_name 


if __name__ == "__main__":
    labels_dir = "annotations_ucfrep/val"
    vid_path = "UCF_Rep/val/v_Biking_g23_c02.mp4"
    traj_file = "point_track/saved_videos/v_Biking_g23_c02/v_Biking_g23_c02_trajectories_frame_22.pkl"
    
    #derive the output name with the video prefix
    vid_name = os.path.basename(vid_path).replace(".mp4", "")
    output_vid_path = f"point_trajectory_counting/{vid_name}_fused_tssm_viz.mp4"
    os.makedirs("point_trajectory_counting", exist_ok=True)
    
    # gt, vid_name = get_count_gt(vid_path, labels_dir)
    # print(f"ground truth: {gt}")
    
    
    #generate the fused matrix data
    tssm, frames = build_fused_tssm(vid_path, traj_file)
    t = len(frames)
    
    # #extract the counting wave from the checkerboard pattern
    # signal_1d = np.mean(tssm, axis=0)
    # signal_1d = medfilt(signal_1d, kernel_size=5)
    # signal_1d -= np.mean(signal_1d)
    # 
    # #find peaks representing the repetitive actions
    # peaks, _ = find_peaks(signal_1d, prominence=np.std(signal_1d)*0.5, distance=10)
    
    #setup the multi-panel visualization
    print(f"generating video: {output_vid_path}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    with imageio.get_writer(output_vid_path, fps=30, codec='libx264', quality=8) as writer:
        for cur_t in range(t):
            ax1.clear(); ax2.clear()
            
            #panel 1: original video frame
            ax1.imshow(frames[cur_t])
            ax1.set_title("input video")
            ax1.axis('off')
            
            #panel 2: the fused self-similarity matrix
            ax2.imshow(tssm, cmap='magma', origin='lower')
            ax2.axvline(cur_t, color='cyan', alpha=0.5)
            ax2.axhline(cur_t, color='cyan', alpha=0.5)
            ax2.set_title("fused rgb + trajectory tssm")
            
            # #panel 3: the 1d rhythm signal and peak count
            # ax3.plot(signal_1d, color='blue', alpha=0.7)
            # ax3.axvline(cur_t, color='red', lw=2)
            # 
            # #highlight peaks crossed so far
            # p_so_far = [p for p in peaks if p <= cur_t]
            # if p_so_far:
            #     ax3.scatter(p_so_far, signal_1d[p_so_far], c='green', s=50, zorder=5)
            # 
            # ax3.set_title(f"rhythm signal | count: {len(p_so_far)}")
            # ax3.set_xlim(0, t)
            # ax3.set_ylim(np.min(signal_1d)*1.2, np.max(signal_1d)*1.2)
            
            #render the frame into the mp4
            canvas.draw()
            buf = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            frame_to_save = buf.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]
            writer.append_data(frame_to_save)
            
            if cur_t % 50 == 0:
                print(f"processed frame {cur_t}/{t}")

    plt.close(fig)
    # print(f"final count for {vid_name}: {len(peaks)}")