import os
import pickle
from sklearn.decomposition import PCA
import numpy as np
from scipy.signal import find_peaks, medfilt
from scipy.fft import fft, fftfreq
import debugpy

#for plotting-
import matplotlib.pyplot as plt
import imageio.v3 as iio
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2

# try:
#     host = "0.0.0.0"
#     port = 5678
#     debugpy.listen((host, port))
#     print(f"[debugpy] Listening on {host}:{port}")

#     print("[debugpy] Waiting for debugger attach...")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"[debugpy] Setup failed: {e}")

def get_periodicity_confidence(signal, fps=30):
    #get fft confidence and dominant frequency
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fps)
    
    amplitude = np.abs(yf)
    
    positive_freqs = xf[:n//2]
    positive_amplitude = amplitude[:n//2]
    #ignore static/constant offset at 0hz
    positive_amplitude[0] = 0 
    
    max_amp = np.max(positive_amplitude)
    mean_amp = np.mean(positive_amplitude)
    
    #confidence is ratio of highest peak to average noise
    confidence = max_amp / (mean_amp + 1e-9)
    dominant_freq = positive_freqs[np.argmax(positive_amplitude)]
    
    return confidence, dominant_freq, positive_freqs, positive_amplitude

def visualize_counting_logic(
    vid_path,
    traj_file,
    output_path="point_trajectory_counting/counting_debug.mp4",
    point_idx=None,
    auto_select_point=True,
    min_motion_std=5.0,
    min_fft_confidence=7.0,
):
    #ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #load data
    frames = iio.imread(vid_path, plugin="FFMPEG")
    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)
    tracks = trajectories['tracks'] 
    fps_video = trajectories.get('fps', 30.0) 
    
    num_points = tracks.shape[2]

    #select point logic
    if point_idx is not None:
        pt_idx = point_idx
    elif auto_select_point:
        pt_idx = -1 #initialize as invalid
        best_conf = -1.0
        for i in range(num_points):
            if np.std(tracks[0, :, i, :]) > min_motion_std:
                p_motion = tracks[0, :, i, :]
                p_vel = np.diff(p_motion, axis=0)
                if len(p_vel) > 0:
                    pca_temp = PCA(n_components=1)
                    v1d_temp = pca_temp.fit_transform(p_vel).flatten()
                    conf, _, _, _ = get_periodicity_confidence(v1d_temp, fps_video)
                    #only update if it meets our strict threshold
                    if conf >= min_fft_confidence and conf > best_conf:
                        best_conf = conf
                        pt_idx = i
        
        #hard fail if no points met the criteria
        if pt_idx == -1:
            raise RuntimeError(
                f"no reliable points found for tracking: no point exceeded fft "
                f"confidence threshold ({min_fft_confidence:.2f})."
            )
    else:
        pt_idx = 0
            
    point_motion = tracks[0, :, pt_idx, :]
    velocity = np.diff(point_motion, axis=0)
    pca = PCA(n_components=1)
    v_1d = pca.fit_transform(velocity).flatten()
    
    conf, dom_freq, freqs, amps = get_periodicity_confidence(v_1d, fps_video)
    
    v_smoothed = medfilt(v_1d, kernel_size=5)
    v_smoothed -= np.mean(v_smoothed)
    
    zero_crossings = np.where(np.diff(np.sign(v_smoothed)))[0]
    refined_crossings = []
    last_idx = -100
    for idx in zero_crossings:
        if idx - last_idx > 5:  
            refined_crossings.append(idx)
            last_idx = idx

    #set target dimensions for ffmpeg compatibility
    target_w, target_h = 1824, 512
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.24, 5.12), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    writer = cv2.VideoWriter(output_path, fourcc, fps_video, (target_w, target_h))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_video, (target_w, target_h))

    print(f"generating visualization for point {pt_idx} (fft conf: {conf:.2f})...")
    
    for t in range(1, len(frames) - 1):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        #left panel
        ax1.imshow(frames[t])
        curr_x, curr_y = point_motion[t]
        ax1.scatter(curr_x, curr_y, c='red', s=40, edgecolors='white')
        ax1.set_title(f"point {pt_idx} tracking")
        ax1.axis('off')
        
        #middle panel
        ax2.plot(v_smoothed, color='blue', alpha=0.3)
        ax2.axhline(0, color='black', lw=1, ls='--')
        ax2.axvline(t, color='red', lw=2) 
        
        crossings_so_far = [c for c in refined_crossings if c <= t]
        if crossings_so_far:
            ax2.scatter(crossings_so_far, [0]*len(crossings_so_far), c='green', zorder=5, s=50)
            
        ax2.set_title(f"1d velocity | reps: {len(crossings_so_far)//2}")
        ax2.set_xlim(0, len(frames))
        ax2.set_ylim(np.min(v_smoothed)*1.2, np.max(v_smoothed)*1.2)
        
        #right panel
        ax3.plot(freqs, amps, color='purple')
        ax3.axvline(dom_freq, color='red', lw=1, ls='--')
        ax3.set_title(f"fft spectrum | conf: {conf:.1f}")
        ax3.set_xlabel("hz")
        ax3.set_ylabel("amp")
        
        canvas.draw()
        vis_frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        vis_frame = vis_frame.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]
        
        bgr_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        bgr_frame = cv2.resize(bgr_frame, (target_w, target_h))
        writer.write(bgr_frame)
        
        if t % 50 == 0:
            print(f"processed frame {t}/{len(frames)}")

    plt.close(fig)
    writer.release()
    print(f"visualization saved to {output_path}")
        
def get_count_gt(vid_path, labels_dir):
    video_name = os.path.splitext(os.path.basename(vid_path))[0]
    video_class = video_name[2:].split("_g", 1)[0]
    
    serach_dir = "annotations_ucfrep/repetition_label"
    parts = None
    
    search_file = video_class
    for root, dirs, files in os.walk(serach_dir):
        for file in files:
            if file == f"{search_file}.txt":
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    parts = content.split(video_name)
                break
                
    #content is now desired file. inside this, search for specific video
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


def count_reps(traj_file, vid_name):
    if not os.path.isfile(traj_file):
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)

    tracks = trajectories['tracks'] 
    fps_video = trajectories.get('fps', 30.0)
    
    num_frames = tracks.shape[1]
    num_points = tracks.shape[2]
    point_counts = [] 
    point_rep_pairs = []
    confidence_threshold = 7.0
    points_over_conf_threshold = 0
    
    for i in range(num_points):
        point_motion = tracks[0, :, i, :] 
        
        point_velocity = np.diff(point_motion, axis=0)

        motion_magnitude = np.linalg.norm(point_velocity, axis=1).mean()
        if motion_magnitude < 0.5: 
            continue

        pca = PCA(n_components=1)
        v_1d = pca.fit_transform(point_velocity).flatten()
        
        #fft confidence check to filter garbage points
        conf, dom_freq, _, _ = get_periodicity_confidence(v_1d, fps_video)
        if conf < confidence_threshold: #heuristic threshold for rhythm confidence
            continue
        points_over_conf_threshold += 1

        v_smoothed = medfilt(v_1d, kernel_size=5)
        v_smoothed -= np.mean(v_smoothed)

        zero_crossings = np.where(np.diff(np.sign(v_smoothed)))[0]

        refined_crossings = []
        if len(zero_crossings) > 0:
            last_idx = -100
            for idx in zero_crossings:
                if idx - last_idx > 5: 
                    refined_crossings.append(idx)
                    last_idx = idx

        reps = len(refined_crossings) // 2
        
        #verify with fft estimation
        #duration in seconds = frames / fps
        duration_sec = num_frames / fps_video
        estimated_reps_fft = dom_freq * duration_sec
        
        #only count if zero-crossing roughly matches fft estimation (within 30% margin)
        #this prevents high-frequency noise points from passing
        if reps > 0 and abs(reps - estimated_reps_fft) / (estimated_reps_fft + 1e-5) < 0.3:
            point_counts.append(reps)
            point_rep_pairs.append((i, reps))
            
    if points_over_conf_threshold == 0:
        raise RuntimeError(
            f"Cannot perform rep counting: no points exceeded FFT confidence threshold ({confidence_threshold:.1f})."
        )

    if not point_counts:
        print(f"Video {vid_name}: No repetitive motion detected.")
        return 0

    final_count = int(np.median(point_counts))
    contributing_points = [point_idx for point_idx, reps in point_rep_pairs if reps == final_count]

    print(f"Video {vid_name}: Detected {final_count} repetitions using {len(point_counts)} active, rhythmic points.")
    print(f"Points contributing to final count ({final_count} reps): {contributing_points}")
    return final_count

if __name__ == "__main__": 
    labels_dir = "annotations_ucfrep/val"
    vid_path = "UCF_Rep/val/v_FrontCrawl_g24_c01.mp4"
    gt, vid_name = get_count_gt(vid_path, labels_dir)
    print(gt)
    
    #now, using the point trajectorties, develop a counting algorithm
    traj_file = "point_track/saved_videos/v_FrontCrawl_g24_c01/v_FrontCrawl_g24_c01_trajectories_frame_22.pkl"
    # count_reps(traj_file, vid_name)
    visualize_counting_logic(vid_path, traj_file, auto_select_point=True)