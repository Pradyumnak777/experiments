import os
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image

print("Loading DINOv2 model...")
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
dino_model.eval()
    
VIDEO_DIR = "videos" 
points_to_sample = 10
grid_size = 20

def get_semantic_mask(img_path, threshold_percentile=60):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    img_tensor = transform(img).unsqueeze(0).cuda()
    
    # get the features
    with torch.no_grad():
        features_dict = dino_model.forward_features(img_tensor)
        features = features_dict['x_norm_patchtokens'] # feats of 14x14 patches in this image(?)
        
    # doing pca to find the "main object"
    features = features.cpu().numpy()[0] 
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features) # so 3 dims
    
    # turn the 1st principal component back into a 2d map/pic
    patch_h, patch_w = 518 // 14, 518 // 14
    foreground_map = pca_features[:, 0].reshape(patch_h, patch_w) 
    
    # normalize
    foreground_map = (foreground_map - foreground_map.min()) / (foreground_map.max() - foreground_map.min())
    
    # resize og dimensions
    foreground_map = torch.tensor(foreground_map).unsqueeze(0).unsqueeze(0)
    foreground_mask_hires = F.interpolate(foreground_map, size=(h, w), mode='bilinear').squeeze().numpy()
    
    # binary mask
    threshold = np.percentile(foreground_mask_hires, threshold_percentile)
    
    #corner background flip.. (can remove?)
    if foreground_mask_hires[0,0] > threshold:
        foreground_mask_hires = 1 - foreground_mask_hires
        threshold = np.percentile(foreground_mask_hires, threshold_percentile)

    binary_mask = foreground_mask_hires > threshold
    return binary_mask


# visualizing binary mask..
# os.makedirs("mask_vis", exist_ok=True)
# mask_path = os.path.join("mask_vis", f"{name}_mask.png")
# mask_img = (active_mask.cpu().numpy().astype(np.uint8) * 255)
# cv2.imwrite(mask_path, mask_img)
# print(f"Saved mask to {mask_path}")

