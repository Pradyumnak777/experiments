import os
import pickle
import numpy as np
import imageio
import imageio.v3 as iio
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, medfilt
from scipy.fft import fft, fftfreq
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

def get_count_gt(vid_path, labels_dir):
    """Retrieves the ground truth count from the UCF-Rep annotation files."""
    video_name = os.path.splitext(os.path.basename(vid_path))[0]
    video_class = video_name[2:].split("_g", 1)[0]
    
    search_dir = os.path.join(labels_dir, "repetition_label")
    parts = None
    
    search_file = video_class
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file == f"{search_file}.txt":
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    if video_name in content:
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

def get_spectral_purity(signal_1d, fps):
    """Calculates the SNR of the dominant frequency, ignoring slow spatial drift."""
    n = len(signal_1d)
    yf = fft(signal_1d)
    xf = fftfreq(n, 1/fps)
    
    amps = np.abs(yf)[:n//2]
    freqs = xf[:n//2]
    
    valid_mask = freqs > 0.5 
    if not np.any(valid_mask):
        return 0.0
        
    valid_amps = amps[valid_mask]
    
    max_amp = np.max(valid_amps)
    mean_amp = np.mean(valid_amps)
    snr = max_amp / (mean_amp + 1e-9)
    
    return snr #high SNR means basically that the point has a consistent oscillation/pattern

def count_reps(traj_file, vid_path, output_path="point_trajectory_counting/thesis_visual_debug.mp4"):
    if not os.path.isfile(traj_file):
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    # 1. Load Data
    frames = iio.imread(vid_path, plugin="FFMPEG")
    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)

    tracks = trajectories['tracks'][0]  # [Time, Points, 2]
    fps_video = trajectories.get('fps', 30.0)
    num_points = tracks.shape[1]
    
    # =====================================================================
    # 2. POINT SELECTION: Energy-Weighted Spectral Purity
    # =====================================================================
    best_pt_idx = 0
    best_score = -1.0
    
    # Store scores for visualization
    all_scores = np.zeros(num_points)
    active_points_mask = np.zeros(num_points, dtype=bool)
    
    print(f"Evaluating {num_points} points for maximum rhythmic energy...")
    for i in range(num_points):
        point_xy = tracks[:, i, :]
        motion_i = np.diff(point_xy, axis=0, prepend=point_xy[:1]) #this is framewise displacement..
        
        #if basically no movementn rhoughout, skip
        if np.mean(np.linalg.norm(motion_i, axis=1)) < 1.0:
            continue
            
        active_points_mask[i] = True
        
        #perform PCA on this
        pca_temp = PCA(n_components=1)
        sig_i = pca_temp.fit_transform(motion_i).flatten()
        sig_i -= np.mean(sig_i) 
        
        snr = get_spectral_purity(sig_i, fps_video)
        energy = np.std(sig_i) #how much a point moves throughout..(on the assumption that )
        final_score = snr * energy  #final score..
        
        all_scores[i] = final_score
        
        if final_score > best_score:
            best_score = final_score
            best_pt_idx = i

    pt_idx = best_pt_idx
    print(f"--> Selected Point {pt_idx} (Kinematic Score: {best_score:.2f})")

    # =====================================================================
    # 3. COUNTING LOGIC: The 4D Kinematic State Vector
    # =====================================================================
    pos = tracks[:, pt_idx, :] 
    # Calculate velocity (frame-to-frame displacement)
    vel = np.diff(pos, axis=0, prepend=pos[:1]) 
    
    # Normalize velocity so X and Y movements carry equal weight
    vel_norm = (vel - np.mean(vel, axis=0)) / (np.std(vel, axis=0) + 1e-6)
    
    # Project 2D velocity down to the 1D "Action Axis"
    pca = PCA(n_components=1)
    signal_1d = pca.fit_transform(vel_norm).flatten()
    
    # Smooth the signal to remove tracker jitter
    signal_1d = medfilt(signal_1d, kernel_size=5)
    signal_1d -= np.mean(signal_1d)
    
    # Count the peaks of the pulses
    std_val = np.std(signal_1d)
    peaks, _ = find_peaks(signal_1d, prominence=std_val * 0.5)
    peak_count = len(peaks)
    
    # =====================================================================
    # 4. THESIS-GRADE VISUALIZATION
    # =====================================================================
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 3 Panels: Video | Signal | Scores
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.0, 5.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    print(f"Generating PCA debug video: {output_path}")
    with imageio.get_writer(output_path, fps=fps_video, codec='libx264', quality=8) as writer:
        for t in range(len(frames)):
            ax1.clear(); ax2.clear(); ax3.clear()

            # --- Panel 1: Video & All Tracked Points ---
            ax1.imshow(frames[t])
            if t < len(tracks): 
                # Plot all active points as tiny background dots
                all_x = tracks[t, active_points_mask, 0]
                all_y = tracks[t, active_points_mask, 1]
                ax1.scatter(all_x, all_y, c='cyan', s=10, alpha=0.3)
                
                # Plot the winning point as a giant red star
                curr_x, curr_y = tracks[t, pt_idx, :]
                ax1.scatter(curr_x, curr_y, c='red', s=150, marker='*', edgecolors='white', linewidths=1.5, zorder=5)
                
            ax1.set_title(f"Tracker Map | Winning Point: {pt_idx}")
            ax1.axis('off')

            # --- Panel 2: 4D Kinematic Signal ---
            ax2.plot(signal_1d, color='blue', alpha=0.4, linewidth=2)
            ax2.axvline(t, color='red', lw=2)
            
            peaks_so_far = [p for p in peaks if p <= t]
            if peaks_so_far:
                ax2.scatter(peaks_so_far, signal_1d[peaks_so_far], c='green', s=60, zorder=5)
            
            ax2.set_title(f"4D Kinematic PCA | Reps: {len(peaks_so_far)}")
            ax2.set_xlim(0, len(frames))
            y_min, y_max = np.min(signal_1d), np.max(signal_1d)
            if y_max > y_min:
                ax2.set_ylim(y_min * 1.2 if y_min < 0 else y_min * 0.8, y_max * 1.2)

            # --- Panel 3: Score Distribution ---
            # Sort scores to make the chart readable
            valid_scores = all_scores[active_points_mask]
            valid_indices = np.where(active_points_mask)[0]
            
            ax3.bar(range(len(valid_scores)), valid_scores, color='gray', alpha=0.5)
            
            # Highlight the winning bar
            winner_bar_idx = np.where(valid_indices == pt_idx)[0][0]
            ax3.bar(winner_bar_idx, valid_scores[winner_bar_idx], color='red')
            
            ax3.set_title("Point Selection Scores (Rhythm × Energy)")
            ax3.set_xlabel("Active Tracker Points")
            ax3.set_ylabel("Kinematic Score")
            ax3.set_xticks([]) # Hide x labels to avoid clutter

            canvas.draw()
            vis_frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            vis_frame = vis_frame.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]
            writer.append_data(vis_frame)

            if t % 50 == 0 and t > 0:
                print(f"Processed frame {t}/{len(frames)}")

    plt.close(fig)
    print(f"Video saved to: {os.path.abspath(output_path)}")
    return peak_count

if __name__ == "__main__": 
    labels_dir = "annotations_ucfrep" 
    vid_path = "UCF_Rep/val/v_BreastStroke_g24_c01.mp4"
    traj_file = "point_track/saved_videos/v_BreastStroke_g24_c01/v_BreastStroke_g24_c01_trajectories_frame_22.pkl"

    gt, vid_name = get_count_gt(vid_path, labels_dir)
    pred = count_reps(traj_file, vid_path)

    print(f"\n--- Results for {vid_name} ---")
    print(f"Ground Truth: {gt}")
    print(f"Predicted:    {pred}")