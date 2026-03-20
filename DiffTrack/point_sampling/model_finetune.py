import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

class DINOv2_LoRA(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base', r=8, lora_alpha=16):
        super(DINOv2_LoRA, self).__init__()
        
        #load the pre-trained backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        #config lora to target the attention layers
        config = LoraConfig(
            r=r, 
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"], 
            lora_dropout=0.1,
            bias="none"
        )
        
        self.model = get_peft_model(self.backbone, config)
        
        #hanging to Conv2d, don't want cross-talk between the 2 different videos
        self.seg_head = nn.Sequential(
            nn.Conv2d(768, 1, kernel_size=1), #1x1 conv looks at each patch independently
            nn.Sigmoid() #output 0-1
        )

    def forward(self, x):
        #x input shape: [b, t, 3, 224, 224]. NOW T=2, there are only 2 image pairs.
        b, t, c, h, w = x.shape
        
        #squash b and t to process frames through dino in one go
        x = x.view(b * t, c, h, w) #x is now [b*t, 3, 224, 224]
        
        #run through lora-wrapped dino
        outputs = self.model(x)
        features = outputs.last_hidden_state #[b*t, 257, 768]
        
        #grab patch tokens (ignore cls token at index 0)
        patch_features = features[:, 1:, :].permute(0, 2, 1).contiguous().view(b * t, 768, 16, 16)
        
        #fix: pass directly into the 2D head so it predicts masks for Video A and Video B independently
        pred_mask_flat = self.seg_head(patch_features) #output- [b*t, 1, 16, 16]
        
        #reshape everything back to [B, T, ...] for the loss functions
        patch_features_unsq = patch_features.view(b, t, 768, 16, 16)
        pred_mask = pred_mask_flat.view(b, t, 1, 16, 16)
        
        return {
            "patch_features": patch_features_unsq,
            "pred_mask": pred_mask
        }
        

def get_robust_mask(flow, depth, threshold_multiplier=1.2, sigma=1.5):
    #flow shape: [b, t, 2, h, w], depth shape: [b, t, 1, h, w]
    b, t, _, h, w = flow.shape
    
    #calculate motion magnitude
    mag = torch.norm(flow, dim=2, keepdim=True) #[b, t, 1, h, w]
        
    #normalizing per frame- f_max: [b, t, 1, 1, 1]
    f_max = depth.flatten(2).max(dim=-1)[0].view(depth.shape[0], depth.shape[1], 1, 1, 1)
    depth_norm = depth / (f_max + 1e-8)
    
    combined_score = mag * depth_norm
    
    #this is perfectly fine for T=2 cross-video!
    #dim=(3,4) calculates the threshold for Video A and Video B completely independently
    mean_score = combined_score.mean(dim=(3, 4), keepdim=True) 
    thresh = torch.clamp(mean_score * threshold_multiplier, min=0.01)
    
    binary_mask = (combined_score > thresh).float()
    
    return binary_mask