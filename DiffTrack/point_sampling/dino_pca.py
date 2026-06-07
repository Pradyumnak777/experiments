import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

VIDEO_NAME  = 'v_BreastStroke_g22_c01'
VIDEO_PATH  = f'UCF_Rep/val/{VIDEO_NAME}.mp4'
OUTPUT_PATH = 'point_sampling/flowfeat_pca.png'

NUM_FRAMES  = 1
SIZE        = (224, 224)


def get_dino_feats(model_name = , device = 'cuda'):
    '''
    getting a tensor of shape- [B, 37, 37, 768] 
    '''
    