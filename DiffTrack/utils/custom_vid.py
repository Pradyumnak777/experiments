import torch
from torch.utils.data import Dataset, dataloader
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F
from typing import Mapping, Tuple, Union
import cv2
import pickle



def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a video (T, H, W, C) to output_size using GPU with torch.

    First resize each frame to (256, 256), then to output_size (H, W) (e.g., 480, 720).
    """
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()

    # First resize to 256x256
    video_resized_256 = F.interpolate(video_tensor, size=(256, 256), mode='bilinear', align_corners=False)

    # Then resize to output size (H, W)
    video_resized_final = F.interpolate(video_resized_256, size=output_size, mode='bilinear', align_corners=False)
    # video_original = F.interpolate(video_tensor, size=(480, 720), mode='bilinear', align_corners=False)
    video_original = video_tensor

    return video_resized_final, video_original


def resize_video_high_resol(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""

    # Then, resize each frame from 256x256 to 720x480:
    video_resized = np.stack([cv2.resize(frame, (output_size[1], output_size[0])) for frame in video])
    return video_resized

class Video(Dataset):
    def __init__(self, args):
        data_root = args.vid_root #this will be a folder, containing folders of videos, which in turn will contain the frames
        self.resize_shape = (args.resize_h, args.resize_w)
        
        # Get all video folders
        self.video_folders = sorted([d for d in os.listdir(data_root) 
                                     if os.path.isdir(os.path.join(data_root, d))])
        self.data_root = data_root
        
        print(f"Found {len(self.video_folders)} videos in {data_root}")
        
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, index):
        
        video_folder = self.video_folders[index]
        video_path = os.path.join(self.data_root, video_folder)
        
        images = sorted(f for f in os.listdir(video_path) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        #open dataroot and store the .jpg files
        frames = []
        for f in images:
            path = os.path.join(video_path, f)
            img = Image.open(path).convert("RGB")
            frames.append(np.array(img))
        
        # # Check if all frames have the same shape
        # shapes = [frame.shape for frame in frames]
        # if len(set(shapes)) > 1:
        #     print(f"ERROR: Inconsistent frame shapes in {video_folder}: {set(shapes)}")
        #     raise ValueError(f"All frames must have the same shape. Found: {set(shapes)}")
        
        video = np.stack(frames, axis = 0)
        frames_ori = None
        if self.resize_shape is not None:
            video, frames_ori = resize_video(video, [self.resize_shape[0], self.resize_shape[1]])
            # video is now a torch tensor (T, C, H, W)
            # For grid sampling, permute to (T, H, W, C)
            # video_for_grid = video.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
        else:
            # video_for_grid = video  # (T, H, W, C)
            frames_ori = torch.from_numpy(video).permute(0, 3, 1, 2).float()

        #now decide on what points to track
        '''
        THIS is challenging...for RAC, knowing what points to track is difficult..
        '''
        
        '''
        grid smapling method below
        '''
        # T = video.shape[0]
        # H = video.shape[2]
        # W = video.shape[3]

        # grid_step = 130  # ~30 points (5×6 grid)
        # y_coords = np.arange(0, H, grid_step, dtype=np.float32)
        # x_coords = np.arange(0, W, grid_step, dtype=np.float32)
        # if y_coords[-1] != H - 1: y_coords = np.append(y_coords, H - 1)
        # if x_coords[-1] != W - 1: x_coords = np.append(x_coords, W - 1)

        # yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

        # query_points = np.stack(
        #     [np.zeros_like(yy, dtype=np.float32), yy, xx], axis=-1
        # ).reshape(-1, 3)
        
        '''
        direction-1: track based on optical flow data..?
        '''
        # points_path = os.path.join("points_to_sample", f"{video_folder}_points.pkl")
        # query_points = None
        # raw_points, ORIG_H, ORIG_W = None, 520, 960
        # if os.path.exists(points_path):
        #     with open(points_path, "rb") as f:
        #         data = pickle.load(f)
            
        #     if isinstance(data, dict):
        #         raw_points = data["points"]
        #         ORIG_H = data["orig_h"]
        #         ORIG_W = data["orig_w"]
                
        # query_points = np.asarray(raw_points, dtype=np.float32)
        
        # target_h = self.resize_shape[0]
        # target_w = self.resize_shape[1]
        # scale_x = target_w / ORIG_W
        # scale_y = target_h / ORIG_H
        
        # query_points[:, 1] *= scale_x
        # query_points[:, 2] *= scale_y
        
        # query_points[:, 1] = np.clip(query_points[:, 1], 0, target_w - 1)
        # query_points[:, 2] = np.clip(query_points[:, 2], 0, target_h - 1)
        
        # return similar to TAPVid format
        '''
        direction 2: using DINOv2..?
        '''
        points_path = os.path.join("points_to_sample", f"{video_folder}_points.pkl")

        query_points = None

        if os.path.exists(points_path):
            try:
                with open(points_path, "rb") as f:
                    data = pickle.load(f)

                raw_points = data["points"]
                # DINO script saves the original dimensions (e.g., 520x960 or whatever the video was)
                ORIG_H = data["orig_h"]
                ORIG_W = data["orig_w"]

                raw_points = np.asarray(raw_points, dtype=np.float32)
                # Stored format from sampler: [t, x, y] -> convert to [t, y, x]
                query_points = raw_points.copy()
                query_points[:, 1] = raw_points[:, 2]  # y
                query_points[:, 2] = raw_points[:, 1]  # x

                # Calculate Scaling Factors to match current resize_shape
                target_h = self.resize_shape[0]
                target_w = self.resize_shape[1]

                scale_y = target_h / ORIG_H
                scale_x = target_w / ORIG_W

                # Apply Scaling: index 1 is y, index 2 is x
                query_points[:, 1] *= scale_y
                query_points[:, 2] *= scale_x

                # Safety Clamp (prevent index out of bounds)
                query_points[:, 1] = np.clip(query_points[:, 1], 0, target_h - 1)
                query_points[:, 2] = np.clip(query_points[:, 2], 0, target_w - 1)

            except Exception as e:
                print(f"Error loading points for {video_folder}: {e}. Falling back to grid.")
                query_points = None

        if query_points is None:
            T = video.shape[0]
            H = video.shape[2]
            W = video.shape[3]
            grid_step = 130  # ~30 points (5×6 grid)
            y_coords = np.arange(0, H, grid_step, dtype=np.float32)
            x_coords = np.arange(0, W, grid_step, dtype=np.float32)
            if y_coords[-1] != H - 1: y_coords = np.append(y_coords, H - 1)
            if x_coords[-1] != W - 1: x_coords = np.append(x_coords, W - 1)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
            query_points = np.stack(
                [np.zeros_like(yy, dtype=np.float32), yy, xx], axis=-1
            ).reshape(-1, 3)
        
        if self.resize_shape is not None:
            frames_tensor = video  # (T, C, H, W) for model input
        else:
            frames_tensor = torch.from_numpy(video).float()  # (T, C, H, W)

        return frames_tensor, query_points, frames_ori


