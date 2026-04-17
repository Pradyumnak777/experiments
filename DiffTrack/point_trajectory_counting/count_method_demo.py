import os
import pickle
from sklearn.decomposition import PCA
import numpy as np
from scipy.signal import find_peaks
import debugpy
from scipy.signal import medfilt

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

def visualize_counting_logic(vid_path, traj_file, output_path="point_trajectory_counting/counting_debug.mp4"):
    # 1. Load data
    frames = iio.imread(vid_path, plugin="FFMPEG")
    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)
    tracks = trajectories['tracks'] 
    
    pt_idx = 40
    for i in range(tracks.shape[2]):
        if np.std(tracks[0, :, i, :]) > 5:
            pt_idx = i
            break
            
    point_motion = tracks[0, :, pt_idx, :]
    velocity = np.diff(point_motion, axis=0)
    pca = PCA(n_components=1)
    v_1d = pca.fit_transform(velocity).flatten()
    v_smoothed = medfilt(v_1d, kernel_size=5)
    v_smoothed -= np.mean(v_smoothed)
    
    zero_crossings = np.where(np.diff(np.sign(v_smoothed)))[0]
    refined_crossings = []
    last_idx = -100
    for idx in zero_crossings:
        if idx - last_idx > 5:  #5 is a heuristic of sorts here..
            refined_crossings.append(idx)
            last_idx = idx

    # 4. Initialize the Figure ONCE (Fixed size for 16x16 macroblocks)
    # 12.8 * 100 = 1280, 4.8 * 100 = 480 (Both divisible by 16)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    fps = 30.0 # Standard framerate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Note: cv2 expects width first, then height. 12.8*100=1280 width, 4.8*100=480 height
    writer = cv2.VideoWriter(output_path, fourcc, fps, (1280, 480))
    print(f"Generating visualization for point {pt_idx}...")
    
    for t in range(1, len(frames) - 1):
        ax1.clear()
        ax2.clear()
        
        # Left: Video Frame + Point
        ax1.imshow(frames[t])
        curr_x, curr_y = point_motion[t]
        ax1.scatter(curr_x, curr_y, c='red', s=40, edgecolors='white')
        ax1.set_title(f"Point {pt_idx} Tracking")
        ax1.axis('off')
        
        # Right: Velocity Signal
        ax2.plot(v_smoothed, color='blue', alpha=0.3)
        ax2.axhline(0, color='black', lw=1, ls='--')
        # Vertical marker for current time
        ax2.axvline(t, color='red', lw=2) 
        
        # Mark detected crossings so far
        crossings_so_far = [c for c in refined_crossings if c <= t]
        if crossings_so_far:
            ax2.scatter(crossings_so_far, [0]*len(crossings_so_far), c='green', zorder=5, s=50)
            
        ax2.set_title(f"1D Velocity | Reps: {len(crossings_so_far)//2}")
        ax2.set_xlim(0, len(frames))
        ax2.set_ylim(np.min(v_smoothed)*1.2, np.max(v_smoothed)*1.2)
        
        # Draw and convert to image
        canvas.draw()
        vis_frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        vis_frame = vis_frame.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]
        
        writer.write(vis_frame)
        
        # Periodically print progress so you know it's not stuck
        if t % 50 == 0:
            print(f"Processed frame {t}/{len(frames)}")

    plt.close(fig)
    writer.release()
    print(f"Visualization saved to {output_path}")
    
    
def get_count_gt(vid_path, labels_dir):
    video_name = os.path.splitext(os.path.basename(vid_path))[0]
    video_class = video_name[2:].split("_g", 1)[0]
    
    serach_dir = "annotations_ucfrep/repetition_label"
    #now search for this exact text file-
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
    #eg- 'v_BabyCrawling_g21_c04'
    
    
    if parts is not None and len(parts) > 1:
        # parts[1] starts with the newline and numbers for our target video
        following_text = parts[1].strip().split('\n\n')[0]
        
        #extract the frame indexes, where the repeating actino occurs
        lines = following_text.splitlines()
        if lines:
            numbers_line = lines[0]
            
            if numbers_line.strip() == "0":
                return 0, video_name
            
            #below is the final ground truth, we are comparing against
            count = len(numbers_line.split())
            return count, video_name

    return 0, video_name # Return 0 if video_name is not found or has no entries


def count_reps(traj_file, vid_name):
    if not os.path.isfile(traj_file):
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    with open(traj_file, "rb") as f:
        trajectories = pickle.load(f)

    tracks = trajectories['tracks'] #this will have x,y info per point, per frame: (1, num_frames, num_points, 2)
    num_frames = tracks.shape[1]
    num_points = tracks.shape[2]
    point_counts = [] #calculating per point.
    
    for i in range(num_points):
        # Extract (T, 2) array for this specific point
        # tracks[0, :, i, 0] is X, tracks[0, :, i, 1] is Y
        point_motion = tracks[0, :, i, :] 
        
        '''
        get point velocity throughout the frames
        NOTE: THIS IS DISPLACEMENT PER FRAME!!
        '''
        point_velocity = np.diff(point_motion, axis=0)
        # point_velocity = []
        # for j in range(len(point_motion)):
        #     if j+1 == len(point_motion):
        #         break
        #     point_velocity.append((point_motion[j+1][0] - point_motion[j][0], point_motion[j+1][1] - point_motion[j][1]))

        #ideally- points with opposing displacements after a specific timeslot would contribute to repitition
        motion_magnitude = np.linalg.norm(point_velocity, axis=1).mean()
        if motion_magnitude < 0.5: # Threshold: less than 0.5 pixels/frame
            continue

        # 4. Project 2D velocity into 1D using PCA
        # This extracts the velocity along the "main line of action"
        pca = PCA(n_components=1)
        v_1d = pca.fit_transform(point_velocity).flatten()

        # 5. Denoising
        # Median filter is great for removing "spikes" while keeping edges sharp
        v_smoothed = medfilt(v_1d, kernel_size=5)
        # Zero-center the signal
        v_smoothed -= np.mean(v_smoothed)

        # 6. Detect Zero-Crossings
        # A zero crossing occurs where sign(v[t]) != sign(v[t+1])
        # We look for sign changes in the smoothed velocity
        zero_crossings = np.where(np.diff(np.sign(v_smoothed)))[0]

        # 7. Physical Constraint: Avoid "flicker" near zero
        # Only count a crossing if there is enough distance between it and the last one
        refined_crossings = []
        if len(zero_crossings) > 0:
            last_idx = -100
            for idx in zero_crossings:
                if idx - last_idx > 5: #NOTE: this is aheauristic! assuming atleast 5 frames between reversal..
                    refined_crossings.append(idx)
                    last_idx = idx

        # Each repetition consists of two reversals (e.g., top and bottom)
        reps = len(refined_crossings) // 2
        
        if reps > 0:
            point_counts.append(reps)
            
    if not point_counts:
        print(f"Video {vid_name}: No repetitive motion detected.")
        return 0

    final_count = int(np.median(point_counts))
    print(f"Video {vid_name}: Detected {final_count} repetitions using {len(point_counts)} active points.")
    return final_count

if __name__ == "__main__": 
    labels_dir = "annotations_ucfrep/val"
    vid_path = "UCF_Rep/val/v_Biking_g21_c01.mp4"
    gt, vid_name = get_count_gt(vid_path, labels_dir)
    print(gt)
    
    #now, using the point trajectorties, develop a counting algorithm
    traj_file = "point_track/saved_videos/v_Biking_g21_c01/v_Biking_g21_c01_trajectories_frame_22.pkl"
    # count_reps(traj_file, vid_name)
    visualize_counting_logic(vid_path, traj_file)