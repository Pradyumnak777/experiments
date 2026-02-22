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
import scipy.sparse.linalg
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # # binary mask
    # threshold = np.percentile(foreground_mask_hires, threshold_percentile)
    
    '''
    UPDATE this heuristic to a more powerful "group vote" method, instead of just relying on the left msot pixel..
    '''
    if foreground_mask_hires[0,0] > 0.5: #a guess...we are assumign tjhat the top left corner is always bacground and SHOULDNT be picked..
        #if it is picked..flip and recalc
        foreground_mask_hires = 1 - foreground_mask_hires
        # threshold = np.percentile(foreground_mask_hires, threshold_percentile)

    # binary_mask = foreground_mask_hires > threshold
    return foreground_mask_hires

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
    # threshold = np.percentile(foreground_mask_hires, threshold_percentile)
    
    if foreground_mask_hires[0,0] > 0.5:
        # flip and redo
        foreground_mask_hires = 1 - foreground_mask_hires
        # threshold = np.percentile(foreground_mask_hires, threshold_percentile)
        
    # binary_mask = foreground_mask_hires > threshold
    return foreground_mask_hires

def cutler_method(img_path, dino_model, patch_size = 14):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    # 518 is for 14, 512 is for 16
    input_res = 518 if patch_size == 14 else 512
    
    transform = T.Compose([
        T.Resize((input_res, input_res)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()
    
    with torch.no_grad():
        if patch_size == 14:
            # dinov2 hub logic
            features_dict = dino_model.forward_features(img_tensor)
            patch_tokens = features_dict['x_norm_patchtokens'][0]
            cls_token = features_dict['x_norm_clstoken'][0]
        else:
            # dinov3 hf logic (skip 1 cls + 4 registers = 5)
            outputs = dino_model(img_tensor)
            patch_tokens = outputs.last_hidden_state[0, 5:, :]
            cls_token = outputs.last_hidden_state[0, 0, :]

    #same as before till here. Now instead of PCA-
    
    '''
    logic of cutLER - read paper..
    '''
    cls_sim = F.cosine_similarity(cls_token.unsqueeze(0), patch_tokens)
    cls_sim = (cls_sim - cls_sim.min()) / (cls_sim.max() - cls_sim.min())
    cls_weight = cls_sim.cpu().numpy()
    
    features = patch_tokens.cpu().numpy()
    A = cosine_similarity(features)
    A = np.where(A < 0, 0, A)
    
    # tau = 0.2 # small threshold to ignore very low attention patches
    tau = np.mean(cls_weight)
    
    #stability work
    A = A * np.maximum(cls_weight[:, np.newaxis] > tau, 1e-6) * np.maximum(cls_weight[np.newaxis, :] > tau, 1e-6)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, A.diagonal() + 1e-6)
    '''
    Ncut/token cut logic..read paper
    '''
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    _, eigvec = scipy.sparse.linalg.eigsh(L, k=2, which='SM', M=D)
    fiedler_vec = eigvec[:, 1]
    fiedler_vec = (fiedler_vec - fiedler_vec.min()) / (fiedler_vec.max() - fiedler_vec.min())
    
    patch_grid = input_res // patch_size
    mask = fiedler_vec.reshape(patch_grid, patch_grid)
    
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    mask_hires = F.interpolate(mask_tensor, size=(h, w), mode='bilinear').squeeze().numpy()
    
    top_mean = np.mean(mask_hires[0, :])
    bottom_mean = np.mean(mask_hires[-1, :])
    left_mean = np.mean(mask_hires[:, 0])
    right_mean = np.mean(mask_hires[:, -1])
    
    edge_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4.0
    
    if edge_mean > 0.5:
        mask_hires = 1 - mask_hires
        
    return mask_hires

def visualize_dino_heatmap(img_path, heatmap, save_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # naming logic
    parts = os.path.normpath(img_path).split(os.sep)
    if "videos" in parts:
        videos_idx = parts.index("videos")
        video_name = parts[videos_idx + 1] if videos_idx + 1 < len(parts) else "unknown"
    else:
        video_name = "unknown"
    frame_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{video_name}_{frame_name}_dino_heatmap.png")

    img = np.array(Image.open(img_path).convert('RGB'))
    
    plt.figure(figsize=(12, 5))
    
    # original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # the heatmap
    plt.subplot(1, 2, 2)
    # but 'viridis' is the classic scientific look
    im = plt.imshow(heatmap, cmap='viridis', vmin=0, vmax=1) 
    plt.title("DINO PCA")
    plt.axis('off')
    
    # adding a colorbar so you can see the scale
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved heatmap to {save_path}")

if __name__ == "__main__":
    img_path = '/scratch/pbk5339/thesis/DiffTrack/videos/benchpress/frame_0011.jpg'
    # mask = dinov3_mask(img_path)
    # save_dir = "dinov2_masks_experiment"
    # visualize_dino_heatmap(img_path, mask, save_dir)
    
    # mask = dinov3_mask(img_path)
    # save_dir = "dinov3_masks_experiment"
    # visualize_dino_heatmap(img_path, mask, save_dir)
    
    #cutler method
    print("Loading DINOv3 for Cutler/TokenCut...")
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    # dino_model = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').cuda()
    dino_model.eval()
    mask = cutler_method(img_path, dino_model, patch_size=14)
    
    save_dir = "cutler_masks_experiment"
    visualize_dino_heatmap(img_path, mask, save_dir)
