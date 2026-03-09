import torch
import torchvision.transforms.v2.functional as F_v2

def get_frame_depth(raw_frame, model, target_size=(224, 224)):
    """
    raw_frame: A single RGB numpy array from cv2
    """
    with torch.no_grad():
        prediction = model.inference([raw_frame])
        
        depth_numpy = prediction.depth[0]
        
        depth_tensor = torch.from_numpy(depth_numpy).unsqueeze(0).unsqueeze(0).float()
        
        depth_tensor = F_v2.resize(depth_tensor, target_size, antialias=True)
        
        d_min = depth_tensor.min()
        d_max = depth_tensor.max()
        depth_tensor = (depth_tensor - d_min) / (d_max - d_min + 1e-8)
        
    # Return as [1, H, W] on the GPU
    return depth_tensor.squeeze(0).cuda()