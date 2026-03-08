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

def maskcut_method(img_path, num_objects=2, patch_size=14, dino_model=None):
    if dino_model is None:
        print("pass the dino model!")
        return -1
    
    device = next(dino_model.parameters()).device
    
    #using hf dinov2 to grab the real attn maps

    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    #518 works for p14
    input_res = 518 
    transform = T.Compose([
        T.Resize((input_res, input_res)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = dino_model(img_tensor, output_attentions=True)
        patch_tokens = outputs.last_hidden_state[0, 1:, :] 
        last_attn = outputs.attentions[-1] 
        cls_attn = last_attn[0, :, 0, 1:].mean(dim=0)
        cls_weight = cls_attn.cpu().numpy()
        
    features = patch_tokens.cpu().numpy()
    
    #calc the similarity matrix between all patches
    A_base = cosine_similarity(features)
    A_base = np.where(A_base < 0, 0, A_base) #no negative
    np.fill_diagonal(A_base, 1.0)
    
    patch_grid = input_res // patch_size
    num_patches = patch_grid * patch_grid
    
    #keep track of what hasn't been picked yet
    available_nodes = np.ones(num_patches, dtype=bool)
    tau = np.mean(cls_weight)
    masks_hires = []
    
    for i in range(num_objects):
        if np.sum(available_nodes) < 10:
            break
            
        if i == 0:
            foreground_mask = cls_weight > tau
        else:
            foreground_mask = available_nodes
            
        current_valid_nodes = foreground_mask & available_nodes
        valid_indices = np.where(current_valid_nodes)[0]
        
        #safety check: if we somehow isolated fewer than 2 patches, stop
        if len(valid_indices) < 2:
            break
            
        # 1. EXTRACT SUB-GRAPH
        A_sub = A_base[np.ix_(valid_indices, valid_indices)]
        np.fill_diagonal(A_sub, 1.0)
        
        # 2. SOLVE ONLY ON SUB-GRAPH
        D_vec_sub = np.sum(A_sub, axis=1)
        D_sub = np.diag(D_vec_sub)
        L_sub = D_sub - A_sub
        
        evals, eigvec = scipy.linalg.eigh(L_sub, D_sub, subset_by_index=[1, 1])
        fiedler_vec_sub = eigvec[:, 0]
        fiedler_vec_sub = (fiedler_vec_sub - fiedler_vec_sub.min()) / (fiedler_vec_sub.max() - fiedler_vec_sub.min() + 1e-8)
        
        # 3. MAP BACK TO FULL GRID
        fiedler_vec_full = np.zeros(num_patches)
        fiedler_vec_full[valid_indices] = fiedler_vec_sub
        
        mask_map = fiedler_vec_full.reshape(patch_grid, patch_grid)
        mask_tensor = torch.tensor(mask_map).unsqueeze(0).unsqueeze(0)
        mask_hires = F.interpolate(mask_tensor, size=(h, w), mode='bilinear').squeeze().numpy()
        
        edge_mean = (np.mean(mask_hires[0, :]) + np.mean(mask_hires[-1, :]) + 
                     np.mean(mask_hires[:, 0]) + np.mean(mask_hires[:, -1])) / 4.0
        
        if edge_mean > 0.5:
            # only invert the valid nodes, leave the background as 0
            fiedler_vec_sub = 1.0 - fiedler_vec_sub
            fiedler_vec_full[valid_indices] = fiedler_vec_sub
            mask_hires = 1.0 - mask_hires
            
        masks_hires.append(mask_hires)
        
        # 4. UPDATE AVAILABLE NODES FOR NEXT OBJECT
        sub_binary_mask = fiedler_vec_sub > np.mean(fiedler_vec_sub)
        full_binary_mask = np.zeros(num_patches, dtype=bool)
        full_binary_mask[valid_indices] = sub_binary_mask
        available_nodes = available_nodes & (~full_binary_mask)

    return masks_hires

def maskcut_tensor_method(img_tensor, num_objects=2, patch_size=14, dino_model=None):
    if dino_model is None:
        print("pass the dino model!")
        return -1
    
    device = next(dino_model.parameters()).device
    
    #using hf dinov2 to grab the real attn maps

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    _, _, h_orig, w_orig = img_tensor.shape
    
    #518 works for p14
    input_res = 518 
    transform = T.Compose([
        T.Resize((input_res, input_res)),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img_tensor).to(device)
    with torch.no_grad():
        outputs = dino_model(img_tensor, output_attentions=True)
        patch_tokens = outputs.last_hidden_state[0, 1:, :] 
        last_attn = outputs.attentions[-1] 
        cls_attn = last_attn[0, :, 0, 1:].mean(dim=0)
        cls_weight = cls_attn.cpu().numpy()
        
    features = patch_tokens.cpu().numpy()
    
    #calc the similarity matrix between all patches
    A_base = cosine_similarity(features)
    A_base = np.where(A_base < 0, 0, A_base) #no negative
    np.fill_diagonal(A_base, 1.0)
    
    patch_grid = input_res // patch_size
    num_patches = patch_grid * patch_grid
    
    #keep track of what hasn't been picked yet
    available_nodes = np.ones(num_patches, dtype=bool)
    tau = np.mean(cls_weight)
    masks_hires = []
    
    for i in range(num_objects):
        if np.sum(available_nodes) < 10:
            break
            
        if i == 0:
            foreground_mask = cls_weight > tau
        else:
            foreground_mask = available_nodes
            
        current_valid_nodes = foreground_mask & available_nodes
        valid_indices = np.where(current_valid_nodes)[0]
        
        #safety check: if we somehow isolated fewer than 2 patches, stop
        if len(valid_indices) < 2:
            break
            
        # 1. EXTRACT SUB-GRAPH
        A_sub = A_base[np.ix_(valid_indices, valid_indices)]
        np.fill_diagonal(A_sub, 1.0)
        
        # 2. SOLVE ONLY ON SUB-GRAPH
        D_vec_sub = np.sum(A_sub, axis=1)
        D_sub = np.diag(D_vec_sub)
        L_sub = D_sub - A_sub
        
        evals, eigvec = scipy.linalg.eigh(L_sub, D_sub, subset_by_index=[1, 1])
        fiedler_vec_sub = eigvec[:, 0]
        fiedler_vec_sub = (fiedler_vec_sub - fiedler_vec_sub.min()) / (fiedler_vec_sub.max() - fiedler_vec_sub.min() + 1e-8)
        
        # 3. MAP BACK TO FULL GRID
        fiedler_vec_full = np.zeros(num_patches)
        fiedler_vec_full[valid_indices] = fiedler_vec_sub
        
        mask_map = fiedler_vec_full.reshape(patch_grid, patch_grid)
        mask_tensor = torch.tensor(mask_map).unsqueeze(0).unsqueeze(0)
        mask_hires = F.interpolate(mask_tensor, size=(h_orig, w_orig), mode='bilinear').squeeze().numpy()
        
        edge_mean = (np.mean(mask_hires[0, :]) + np.mean(mask_hires[-1, :]) + 
                     np.mean(mask_hires[:, 0]) + np.mean(mask_hires[:, -1])) / 4.0
        
        if edge_mean > 0.5:
            # only invert the valid nodes, leave the background as 0
            fiedler_vec_sub = 1.0 - fiedler_vec_sub
            fiedler_vec_full[valid_indices] = fiedler_vec_sub
            mask_hires = 1.0 - mask_hires
            
        masks_hires.append(mask_hires)
        
        # 4. UPDATE AVAILABLE NODES FOR NEXT OBJECT
        sub_binary_mask = fiedler_vec_sub > np.mean(fiedler_vec_sub)
        full_binary_mask = np.zeros(num_patches, dtype=bool)
        full_binary_mask[valid_indices] = sub_binary_mask
        available_nodes = available_nodes & (~full_binary_mask)

    return masks_hires


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


def visualize_maskcut_results(img_path, masks, save_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    #naming logic remains the same
    parts = os.path.normpath(img_path).split(os.sep)
    video_name = parts[parts.index("videos") + 1] if "videos" in parts else "unknown"
    frame_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    
    img = np.array(Image.open(img_path).convert('RGB'))
    num_masks = len(masks)
    
    #create a grid: 1 for original + N for masks
    plt.figure(figsize=(4 * (num_masks + 1), 5))
    
    #show original
    plt.subplot(1, num_masks + 1, 1)
    plt.imshow(img)
    plt.title("original")
    plt.axis('off')
    
    #show each object found by maskcut
    for i, m in enumerate(masks):
        plt.subplot(1, num_masks + 1, i + 2)
        plt.imshow(m, cmap='viridis', vmin=0, vmax=1)
        plt.title(f"object {i+1}")
        plt.axis('off')
    
    save_path = os.path.join(save_dir, f"{video_name}_{frame_name}_maskcut.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"saved multi-object visualization to {save_path}")

if __name__ == "__main__":
    img_path = '/scratch/pbk5339/thesis/DiffTrack/videos/swim/frames_009.jpg'
    # mask = dinov3_mask(img_path)
    # save_dir = "dinov2_masks_experiment"
    # visualize_dino_heatmap(img_path, mask, save_dir)
    
    # mask = dinov3_mask(img_path)
    # save_dir = "dinov3_masks_experiment"
    # visualize_dino_heatmap(img_path, mask, save_dir)
    
    #cmaskcututler method
    print("Loading DINOv3 for maskcut/cutler...")
    # dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    # # dino_model = AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m').cuda()
    # dino_model.eval()
    model_name = 'facebook/dinov2-small'
    dino_model = AutoModel.from_pretrained(model_name, output_attentions=True).cuda()
    dino_model.eval()

    
    masks = maskcut_method(img_path, num_objects = 2, dino_model=dino_model)
    
    save_dir = "cutler_masks_experiment"
    visualize_maskcut_results(img_path, masks, save_dir)
