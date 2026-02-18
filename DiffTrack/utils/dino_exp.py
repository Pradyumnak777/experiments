import os
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import AutoModel
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))

def dinov2_mask(img_path, threshold_percentile=50):
    print("Loading DINOv2 model...")
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    dino_model.eval()
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
    foreground_mask_hires = F.interpolate(foreground_map, size=(h, w), mode='bilinear').squeeze().numpy() #converting back to old size
    
    # binary mask
    threshold = np.percentile(foreground_mask_hires, threshold_percentile)
    
    '''
    UPDATE this heuristic to a more powerful "group vote" method, instead of just relying on the left msot pixel..
    '''
    if foreground_mask_hires[0,0] > threshold: #a guess...we are assumign tjhat the top left corner is always bacground and SHOULDNT be picked..
        #if it is picked..flip and recalc
        foreground_mask_hires = 1 - foreground_mask_hires
        threshold = np.percentile(foreground_mask_hires, threshold_percentile)

    binary_mask = foreground_mask_hires > threshold
    return binary_mask

def dinov3_mask(img_path, threshold_percentile=60):
    print("Loading DINOv3 model...")
    # using the 'base' model with patch size 16. 
    dino_model = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').cuda()
    dino_model.eval()
    
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    # dinov3 uses patch size 16, so 512x512 divides evenly (32x32 patches)
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    img_tensor = transform(img).unsqueeze(0).cuda()
    
    # get the features
    with torch.no_grad():
        outputs = dino_model(img_tensor)
        # dinov3 structure: [CLS, REG1, REG2, REG3, REG4, PATCHES...]
        # so we skip the first 5 tokens to get just the image patches
        features = outputs.last_hidden_state[:, 5:, :] 
        
    # doing pca to find the "main object"
    features = features.cpu().numpy()[0] 
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features) 
    
    # turn the 1st principal component back into a 2d map/pic
    # 512 / 16 = 32
    patch_h, patch_w = 512 // 16, 512 // 16
    foreground_map = pca_features[:, 0].reshape(patch_h, patch_w) 
    
    # normalize
    foreground_map = (foreground_map - foreground_map.min()) / (foreground_map.max() - foreground_map.min())
    
    # resize og dimensions
    foreground_map = torch.tensor(foreground_map).unsqueeze(0).unsqueeze(0)
    foreground_mask_hires = F.interpolate(foreground_map, size=(h, w), mode='bilinear').squeeze().numpy() 
    
    # binary mask
    threshold = np.percentile(foreground_mask_hires, threshold_percentile)
    
    if foreground_mask_hires[0,0] > threshold:
        # flip and redo
        foreground_mask_hires = 1 - foreground_mask_hires
        threshold = np.percentile(foreground_mask_hires, threshold_percentile)
        
    binary_mask = foreground_mask_hires > threshold
    return binary_mask


def visualize_dino_mask(img_path, mask, save_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # Naming logic
    parts = os.path.normpath(img_path).split(os.sep)
    if "videos" in parts:
        videos_idx = parts.index("videos")
        video_name = parts[videos_idx + 1] if videos_idx + 1 < len(parts) else "unknown"
    else:
        video_name = "unknown"
    frame_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{video_name}_{frame_name}_dino_mask.png")

    img = np.array(Image.open(img_path).convert('RGB'))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='viridis')
    plt.title("DINO Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved mask visualization to {save_path}")


if __name__ == "__main__":
    img_path = '/scratch/pbk5339/thesis/DiffTrack/videos/swim_2/frames_001.jpg'
    mask = dinov3_mask(img_path)
    save_dir = "dinov3_masks_experiment"
    visualize_dino_mask(img_path, mask, save_dir)
