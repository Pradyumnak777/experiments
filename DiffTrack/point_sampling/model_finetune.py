import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

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
        
        self.model = get_peft_model(self.backbone, config)
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(768, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w) 
        
        #fix: we no longer need to pass output_attentions=True here
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
            "last_attn": last_layer_attn 
        }

def get_robust_mask(flow, depth, threshold_multiplier=1.2, flow_weight=0.5, sigma=1.5):
    b, t, _, h, w = flow.shape
    
    median_flow = flow.view(b, t, 2, -1).median(dim=3, keepdim=True)[0].view(b, t, 2, 1, 1)
    relative_flow = flow - median_flow
    
    mag = torch.norm(relative_flow, dim=2, keepdim=True)
    squashed_mag = torch.sqrt(mag + 1e-8)
        
    f_max = depth.flatten(2).max(dim=-1)[0].view(depth.shape[0], depth.shape[1], 1, 1, 1)
    depth_norm = depth / (f_max + 1e-8)
    
    # y = torch.linspace(-1, 1, h, device=flow.device).view(1, 1, 1, h, 1)
    # x = torch.linspace(-1, 1, w, device=flow.device).view(1, 1, 1, 1, w)
    # center_prior = torch.exp(-(x**2 + y**2) / (2 * 0.7**2))
    
    combined_score = (squashed_mag ** flow_weight) * depth_norm
    
    mean_score = combined_score.mean(dim=(3, 4), keepdim=True) 
    thresh = torch.clamp(mean_score * threshold_multiplier, min=0.01)
    binary_mask = (combined_score > thresh).float()
    
    return binary_mask