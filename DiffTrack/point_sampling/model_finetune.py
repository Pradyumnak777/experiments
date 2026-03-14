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
        #this lets dino learn which parts of the frame matter for motion
        config = LoraConfig(
            r=r, 
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"], 
            lora_dropout=0.1,
            bias="none"
        )
        
        #wrap the model so we only train the tiny lora adapters
        self.model = get_peft_model(self.backbone, config)

    def forward(self, x):
        #x input shape: [b, t, 3, 224, 224]
        b, t, c, h, w = x.shape
        
        #squash b and t to process frames through dino in one go
        x = x.view(b * t, c, h, w)
        
        #run through lora-wrapped dino
        outputs = self.model(x)
        features = outputs.last_hidden_state #[b*t, 257, 768]
        
        #grab patch tokens (ignore cls token at index 0)
        #reshape back to [b*t, 768, 16, 16] for spatial masking later
        patch_features = features[:, 1:, :].permute(0, 2, 1).view(b * t, 768, 16, 16)
        
        #separate batch and time again before returning
        return patch_features.view(b, t, 768, 16, 16)

def get_robust_mask(flow, depth, threshold_multiplier=1.2):
    #flow shape: [b, t, 2, h, w], depth shape: [b, t, 1, h, w]
    #this acts as a noisy teacher, giving dino a hint on where to look
    
    # calculate motion magnitude
    mag = torch.norm(flow, dim=2, keepdim=True) #[b, t, 1, h, w]
    
    #normalize depth to 0-1 range
    depth_norm = depth / (depth.max() + 1e-8)
    
    combined_score = mag * depth_norm
    
    #instead of a hard cutoff, we only take pixels 'more active' than the average
    #this saves us if the whole frame is panning
    mean_score = combined_score.mean(dim=(3, 4), keepdim=True)
    binary_mask = (combined_score > (mean_score * threshold_multiplier)).float()
    
    return binary_mask