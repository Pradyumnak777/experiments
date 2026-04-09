import cv2
import pickle
import torch
import numpy as np
import imageio.v3 as iio
from matplotlib import cm

#paths to the data i generated
vid_1_info = "point_track/saved_videos/v_BodyWeightSquats_g21_c04/v_BodyWeightSquats_g21_c04_trajectories_frame_22.pkl"
vid_2_info = "point_track/saved_videos/v_BodyWeightSquats_g22_c03/v_BodyWeightSquats_g22_c03_trajectories_frame_22.pkl"
mapping_path = "point_track/saved_paths/test1.pkl"

#load the mapping and the track data
with open(vid_1_info, "rb") as f: data_1 = pickle.load(f)
with open(vid_2_info, "rb") as f: data_2 = pickle.load(f)
with open(mapping_path, "rb") as f: mapping = pickle.load(f)

#load the specific frames used for matching
img_1 = iio.imread(data_1["video_path"], plugin="FFMPEG")[data_1["query_frame"]]
img_2 = iio.imread(data_2["video_path"], plugin="FFMPEG")[data_2["query_frame"]]

#i want them to be the same height for the side-by-side view
h1, w1 = img_1.shape[:2]
h2, w2 = img_2.shape[:2]
target_h = max(h1, h2)
img_1 = cv2.resize(img_1, (int(w1 * (target_h/h1)), target_h))
img_2 = cv2.resize(img_2, (int(w2 * (target_h/h2)), target_h))

#create the canvas
canvas = np.hstack([img_1, img_2])
offset_x = img_1.shape[1]

#i'll use a rainbow colormap to distinguish the 100 points
num_points = len(mapping)
colors = (cm.rainbow(np.linspace(0, 1, num_points))[:, :3] * 255).astype(np.uint8)

#loop through the mapping to draw the points
for i, (idx_1, idx_2) in enumerate(mapping.items()):
    #grab raw points and scale them if the image was resized
    p1 = data_1["tracks"][0, data_1["query_frame"], idx_1]
    p2 = data_2["tracks"][0, data_2["query_frame"], idx_2]
    
    #scale coords to match the resized canvas
    c1 = (int(p1[0] * (img_1.shape[1]/w1)), int(p1[1] * (target_h/h1)))
    c2 = (int(p2[0] * (img_2.shape[1]/w2)) + offset_x, int(p2[1] * (target_h/h2)))
    
    color = tuple(map(int, colors[i]))
    
    #draw the points on both sides
    cv2.circle(canvas, c1, 4, color, -1)
    cv2.circle(canvas, c2, 4, color, -1)
    
    cv2.line(canvas, c1, c2, color, 1, cv2.LINE_AA)

#save the diagnostic image
out_path = "point_track/saved_paths/semantic_match_verification.png"
cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print(f"saved visualization to {out_path}")