import os
import pickle
import numpy as np
import imageio
import imageio.v3 as iio
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, medfilt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

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

def count_reps(traj_file, vid_path, output_path="point_trajectory_counting/pca_clean_debug.mp4"):
    if not os.path.isfile(traj_file):
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    # 1. Load Data
    frames = iio.imread(vid_path, plugin="FFMPEG")
    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)

    tracks = trajectories['tracks'][0]  # [Time, Points, 2]
    fps_video = trajectories.get('fps', 30.0)
    
    # 2. PCA Core Logic
    # Using Point 0 as the reference point
    pt_idx = 0
    motion = tracks[:, pt_idx, :]  # [Time, 2]
    
    # Project 2D (X,Y) motion into the 1D dominant axis
    pca = PCA(n_components=1)
    signal_1d = pca.fit_transform(motion).flatten()
    
    # Pre-process: Clean jitter and center the signal
    signal_1d = medfilt(signal_1d, kernel_size=5)
    signal_1d -= np.mean(signal_1d)
    
    # 3. Robust Peak Counting (Prominence-based)
    std_val = np.std(signal_1d)
    peaks, _ = find_peaks(signal_1d, prominence=std_val * 0.5)
    peak_count = len(peaks)
    
    # 4. Visualization
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    print(f"Generating PCA debug video: {output_path}")
    with imageio.get_writer(output_path, fps=fps_video, codec='libx264', quality=8) as writer:
        for t in range(len(frames)):
            ax1.clear()
            ax2.clear()

            # Left: Video + Point Tracking
            ax1.imshow(frames[t])
            if t < len(motion):
                curr_x, curr_y = motion[t]
                ax1.scatter(curr_x, curr_y, c='red', s=40, edgecolors='white')
            ax1.set_title(f"Tracking Point {pt_idx}")
            ax1.axis('off')

            # Right: 1D PCA Waveform
            ax2.plot(signal_1d, color='blue', alpha=0.3)
            ax2.axvline(t, color='red', lw=2)
            
            peaks_so_far = [p for p in peaks if p <= t]
            if peaks_so_far:
                ax2.scatter(peaks_so_far, signal_1d[peaks_so_far], c='green', zorder=5)
            
            ax2.set_title(f"PCA Projections | Reps: {len(peaks_so_far)}")
            ax2.set_xlim(0, len(frames))
            ax2.set_ylim(np.min(signal_1d) * 1.2, np.max(signal_1d) * 1.2)

            canvas.draw()
            vis_frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            vis_frame = vis_frame.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]
            writer.append_data(vis_frame)

    plt.close(fig)
    return peak_count

if __name__ == "__main__": 
    labels_dir = "annotations_ucfrep/val"
    vid_path = "UCF_Rep/val/v_Biking_g21_c01.mp4"
    traj_file = "point_track/saved_videos/v_Biking_g21_c01/v_Biking_g21_c01_trajectories_frame_22.pkl"

    # Get Ground Truth
    gt, vid_name = get_count_gt(vid_path, labels_dir)
    
    # Run Prediction
    pred = count_reps(traj_file, vid_path)

    print(f"\n--- Results for {vid_name} ---")
    print(f"Ground Truth: {gt}")
    print(f"Predicted:    {pred}")