import os
import pickle
import numpy as np
import imageio
import imageio.v3 as iio
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, medfilt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


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

def get_spectral_purity(signal_1d, fps):
    #calculate how clean the rhythm is
    n = len(signal_1d)
    yf = fft(signal_1d)
    xf = fftfreq(n, 1/fps)
    amps = np.abs(yf)[:n//2]
    freqs = xf[:n//2]
    
    #ignore slow camera drift
    valid_mask = (freqs > 0.2) & (freqs < 5.0) #jittering frequencies..
    if not np.any(valid_mask): return 0.0
    
    valid_amps = amps[valid_mask]
    snr = np.max(valid_amps) / (np.mean(valid_amps) + 1e-9)
    return snr

def count_reps(traj_file, vid_path, output_path="point_trajectory_counting/pca_clean_debug.mp4"):
    if not os.path.isfile(traj_file):
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    # 1. Load Data
    frames = iio.imread(vid_path, plugin="FFMPEG")
    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)

    tracks = trajectories['tracks'][0]  # [Time, Points, 2]
    fps_video = trajectories.get('fps', 30.0)
    num_pts = tracks.shape[1]

    '''
    NOTE: the method is to find the "arc/path" of the point. If a point is good,
    its traced "path" should remain consistent/overlap, invariant of scale or its absolute location..
    '''
    best_pt_idx = 0
    best_score = -1.0
    all_scores = np.zeros(num_pts)
    active_mask = np.zeros(num_pts, dtype=bool) #mask of selected points (?)
    
    print(f"analyzing {num_pts} points in vector space...")
    for i in range(num_pts):
        pos_i = tracks[:, i, :]
        vel_i = np.diff(pos_i, axis=0, prepend=pos_i[:1])
        
        #normalize to unit vectors..as we want it be scale invariant..
        mags = np.linalg.norm(vel_i, axis=1, keepdims=True)
        unit_vel = vel_i / (mags + 1e-6)
        
        # some points may move very less/jitter
        if np.mean(mags) < 1.0: 
            continue
        
        active_mask[i] = True #select this point
        
        #project 2d unit path to 1d rhythm
        pca_temp = PCA(n_components=1)
        sig_i = pca_temp.fit_transform(unit_vel).flatten()
        
        #scoring part
        snr = get_spectral_purity(sig_i, fps_video) #high SNR means good periodicitiy..
        energy = np.std(mags) #use raw magnitude for energy weight...
        score = snr * energy #scoring part: 
        
        all_scores[i] = score
        if score > best_score:
            best_score = score
            best_pt_idx = i
    
    #selecting best point      
    pt_idx = best_pt_idx
    print(f"selected point {pt_idx} with score {best_score:.2f}")
    
    final_pos = tracks[:, pt_idx, :]
    final_vel = np.diff(final_pos, axis=0, prepend=final_pos[:1])
    
    # final_mags = np.linalg.norm(final_vel, axis=1, keepdims=True)
    # final_unit_vel = final_vel / (final_mags + 1e-6)
    
    #project to 1D now
    pca = PCA(n_components=1)
    signal_1d = pca.fit_transform(final_vel).flatten()
    
    #smoothening
    signal_1d = medfilt(signal_1d, kernel_size=5)
    signal_1d -= np.mean(signal_1d)
    
    #perform counting
    std_val = np.std(signal_1d)
    peaks, _ = find_peaks(signal_1d, prominence=std_val * 0.5, height=0, distance=5) #height=0, so counting positive side peaks only..
    #distance = 3, means repitiotns canot relaistically occur every 3 frames, as that is too small..
    pred_count = len(peaks)
    
    '''
    #NOTE: adding some visualization..
    '''
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    output_dir = "point_trajectory_counting"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{vid_name}_visualization.mp4")

    #3 panels: tracker map | unit vector rhythm | point scores
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.0, 5.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    print(f"generating vector space debug video: {output_path}")
    with imageio.get_writer(output_path, fps=fps_video, codec='libx264', quality=8) as writer:
        for t in range(len(frames)):
            ax1.clear(); ax2.clear(); ax3.clear()

            #panel 1: video & tracked points
            ax1.imshow(frames[t])
            if t < len(tracks): 
                #plot all moving points as tiny background dots
                all_x = tracks[t, active_mask, 0]
                all_y = tracks[t, active_mask, 1]
                ax1.scatter(all_x, all_y, c='cyan', s=10, alpha=0.3)
                
                #giant red star for the winning point
                curr_x, curr_y = tracks[t, pt_idx, :]
                ax1.scatter(curr_x, curr_y, c='red', s=150, marker='*', edgecolors='white', linewidths=1.5, zorder=5)
                
            ax1.set_title(f"tracker map | winning point: {pt_idx}")
            ax1.axis('off')

            #panel 2: the unit vector rhythm (1d pca)
            ax2.plot(signal_1d, color='blue', alpha=0.6, linewidth=2)
            ax2.axvline(t, color='red', lw=2)
            
            #show peaks counted so far
            peaks_so_far = [p for p in peaks if p <= t]
            if peaks_so_far:
                ax2.scatter(peaks_so_far, signal_1d[peaks_so_far], c='green', s=60, zorder=5)
            
            ax2.set_title(f"unit vector pca | reps: {len(peaks_so_far)}")
            ax2.set_xlim(0, len(frames))
            
            #keep y-axis stable
            y_min, y_max = np.min(signal_1d), np.max(signal_1d)
            if y_max > y_min:
                ax2.set_ylim(y_min * 1.2 if y_min < 0 else y_min * 0.8, y_max * 1.2)

            valid_scores = all_scores[active_mask]
            valid_indices = np.where(active_mask)[0]
            
            ax3.bar(range(len(valid_scores)), valid_scores, color='gray', alpha=0.5)
            
            #highlight why the red star won
            winner_bar_idx = np.where(valid_indices == pt_idx)[0][0]
            ax3.bar(winner_bar_idx, valid_scores[winner_bar_idx], color='red')
            
            ax3.set_title("point scores (snr * energy)")
            ax3.set_xlabel("active points")
            ax3.set_xticks([]) #hide messy x-labels

            canvas.draw()
            vis_frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            vis_frame = vis_frame.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]
            writer.append_data(vis_frame)

            if t % 50 == 0 and t > 0:
                print(f"processed frame {t}/{len(frames)}")

    plt.close(fig)
    print(f"video saved to: {os.path.abspath(output_path)}")
    
    '''
    NOTE: back to normal execution..
    '''
    
    return pred_count
    
    
if __name__ == "__main__": 
    labels_dir = "annotations_ucfrep/val"
    vid_path = "UCF_Rep/val/v_Rowing_g22_c04.mp4"
    traj_file = "point_track/saved_videos/v_Rowing_g22_c04/v_Rowing_g22_c04_trajectories_frame_22.pkl"

    # Get Ground Truth
    gt, vid_name = get_count_gt(vid_path, labels_dir)
    
    # Run Prediction
    pred = count_reps(traj_file, vid_path)

    print(f"\nResults for {vid_name}: ")
    print(f"Ground Truth: {gt}")
    print(f"Predicted:    {pred}")