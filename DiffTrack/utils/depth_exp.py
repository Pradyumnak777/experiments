import torch
from depth_anything_3.api import DepthAnything3
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
model = model.to(device=device)


def get_frame_depth(frame_tensor, target_size=(224, 224)):
    with torch.no_grad():
        depth = model(frame_tensor)
        depth = F.interpolate(depth.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False)
        d_min = depth.min()
        d_max = depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-8)
    
    return depth.squeeze(1)