import torch
import torchvision.transforms.v2.functional as F_v2
import os
from contextlib import redirect_stdout

def get_batch_depth(raw_frames_list, model, target_size=(224, 224), chunk_size=16):
    #chunk size 16 is safer for base model
    all_depths = []
    
    with torch.no_grad():
        for i in range(0, len(raw_frames_list), chunk_size):
            chunk = raw_frames_list[i : i + chunk_size]
            
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                predictions = model.inference(chunk)
            
            depth_numpy = predictions.depth
            depth_tensor = torch.from_numpy(depth_numpy).unsqueeze(1).float()
            depth_tensor = F_v2.resize(depth_tensor, target_size, antialias=True)
            
            #normalize 0-1 per frame
            b = depth_tensor.shape[0]
            depth_flat = depth_tensor.view(b, -1)
            d_min = depth_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            d_max = depth_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            depth_tensor = (depth_tensor - d_min) / (d_max - d_min + 1e-8)
            
            all_depths.append(depth_tensor.cpu().half())
            
            #clear vram after every chunk
            torch.cuda.empty_cache()
            
    return torch.cat(all_depths, dim=0)