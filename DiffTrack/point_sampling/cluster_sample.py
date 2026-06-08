'''
1. take input video
2. perform flowfeat to get 128 dim emebedding for every pixel, in every frame
3. perform "clustering"(?), to sample points
'''

import cv2
from flowfeat_mask import load_flowfeat, get_flowfeat  #will return the 128 dim embeddings
from flowfeat_mask import load_video_frames #returns [1, t, 3, h, w]
import torch

VIDEO_NAME  = 'v_Biking_g21_c01'
VIDEO_PATH  = f'UCF_Rep/val/{VIDEO_NAME}.mp4'




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading FlowFeat...")
    model = load_flowfeat(device=device)

    print(f"Loading video: {VIDEO_PATH}")
    pixels = load_video_frames(VIDEO_PATH).to(device)
    
    print("getting FlowFeat features...")
    feats = get_flowfeat(pixels, model)     #[1, T, 128, 224, 224]
    
    #now, clustering needs to happen
    