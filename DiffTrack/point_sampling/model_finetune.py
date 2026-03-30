import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import numpy as np

class DINOv2_LoRA(nn.Module):
    def __init__(self, model_name='facebook/dinov2-with-registers-base', r=8, lora_alpha=16):
        super(DINOv2_LoRA, self).__init__()
        
        self.backbone = AutoModel.from_pretrained(model_name, output_attentions=True)
        
        config = LoraConfig(
            r=r, 
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"], 
            lora_dropout=0.1,
            bias="none"
        )
        
        self.model = get_peft_model(self.backbone, config) #searches dino for Q,K,V matrices and injects adaptation..
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(768, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w) 
        
        outputs = self.model(x)
        features = outputs.last_hidden_state 
        
        #skip 1 cls token + 4 register tokens
        patch_features = features[:, 5:, :].permute(0, 2, 1).contiguous().view(b * t, 768, 16, 16)
        pred_mask_flat = self.seg_head(patch_features) 
        
        patch_features_unsq = patch_features.view(b, t, 768, 16, 16)
        pred_mask = pred_mask_flat.view(b, t, 1, 16, 16)
        
        #this will now safely grab the 12th layer's attention
        last_layer_attn = outputs.attentions[-1]
        
        return {
            "patch_features": patch_features_unsq,
            "pred_mask": pred_mask,
            "last_attn": last_layer_attn  #not being used..!
        }

def get_robust_mask(flow, threshold_multiplier=1.2, flow_weight=0.6): #lower weight and also check??
    b, t, _, h, w = flow.shape
    device = flow.device
    # depth = depth.squeeze(2) # [b, t, h, w]

    #median flow, then calculate relative flow
    median_flow = flow.view(b, t, 2, -1).median(dim=3, keepdim=True)[0].view(b, t, 2, 1, 1) #median for both u and v..
    rel_flow = flow - median_flow # "relative" flow..
    
    
    rel_mag = torch.norm(rel_flow, dim=2) # [b, t, h, w]
    # atan2(v, u) gives the relative direction of motion
    rel_angle = torch.atan2(rel_flow[:, :, 1], rel_flow[:, :, 0])  #get the a ngle wrt positive x axis (by default)

    #this is a heauristic..
    '''
    Angle differences between frames. If the angle stays consistent between these 3 consecutive frames, then it is given
    a "high score". If there is chaotic jitter every few frames, it is likely not an action..
    '''
    angle_diff_t = torch.abs(rel_angle[:, 1:] - rel_angle[:, :-1])
    # Handle the pi/-pi wrap around (hapens in 360 deg rotations..)
    angle_diff_t = torch.where(angle_diff_t > np.pi, 2*np.pi - angle_diff_t, angle_diff_t)
    
    angle_consistency = torch.cos(angle_diff_t) # High for small angle changes
    angle_consistency = F.pad(angle_consistency, (0,0,0,0,1,0), value=1.0) #pad extra frame, as only 2 difference frames are obtained
    angle_consistency = torch.clamp(angle_consistency, min=0.1)

    #
    smooth_mag = rel_mag.mean(dim=1, keepdim=True).expand(-1, t, -1, -1) #avg magnitued over the frames
    mag_norm = smooth_mag / (smooth_mag.view(b, t, -1).max(dim=-1)[0].view(b, t, 1, 1) + 1e-8) #normalize
    
    #score
    score = (mag_norm ** flow_weight) * (angle_consistency ** 2)
    
    #threshold
    mean_score = score.mean(dim=(2, 3), keepdim=True)
    thresh = torch.clamp(mean_score * 1.4, min=0.06) #anything b/w mean*1.4 and 0.06
    binary_mask = (score > thresh).float()
    
    #border edges..black borders..
    '''
    alternative- recognize. by pixelc olor? pitch black at border..ignore?
    '''
    border_h = int(h * 0.05)
    border_w = int(w * 0.05)
    spatial_mask = torch.ones_like(binary_mask)
    spatial_mask[:, :, :border_h, :] = 0
    spatial_mask[:, :, -border_h:, :] = 0
    spatial_mask[:, :, :, :border_w] = 0
    spatial_mask[:, :, :, -border_w:] = 0

    binary_mask = binary_mask * spatial_mask

    return binary_mask.unsqueeze(2)