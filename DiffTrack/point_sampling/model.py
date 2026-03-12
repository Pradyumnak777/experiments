import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RepMask(nn.Module):
    def __init__(self, embed_dim = 128): #here val_dim is flow(2) + depth(1) 
        super(RepMask, self).__init__()
        
        self.proj_q = nn.Linear(384, embed_dim)  #this is dino
        self.proj_kv = nn.Linear(3, embed_dim) # (flow + depth)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        self.proj_out = nn.Linear(embed_dim, 387) #output from attn_layer is embed_dim dimension, this is for projecting back
        
        
    def forward(self, x):
        # x input: [B(batch), 387(384+2+1), T, H, W]
        b, c, t, h, w = x.shape
        summary = torch.mean(x, dim=(2, 3, 4)) #HOW EFFECTIVE IS THIS??(check online..)
        dino_part = summary[:, :384]
        phys_part = summary[:, 384:]
        
        Q = self.proj_q(dino_part).unsqueeze(1) #[B, 1, 384] -> [B, 1, 128]
        KV = self.proj_kv(phys_part).unsqueeze(1) #[B, 1, 3] -> [B 1, 128]
        
        attn_out, weights = self.cross_attn(query=Q, key=KV, value=KV)
        
        channel_weights = self.proj_out(attn_out.squeeze(1)) #[][B, 387]
        channel_weights = torch.sigmoid(channel_weights).view(b, 387, 1, 1, 1)
        
        return x * channel_weights, weights
        

class MaskGen(nn.Module):
    def __init__(self):
        super(MaskGen, self).__init__()
        
        self.attention_block = RepMask(embed_dim=128)
        
        self.temporal_conv = nn.Conv3d(in_channels=387, out_channels=64, kernel_size=(8, 1, 1)) #collapsing the clip of 387 channels
        self.dec_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1) #this is the final mask
        
    def forward(self, x):
        #get clean features using cross attention above
        cleaned_x, attn_weights = self.attention_block(x)
        
        # new output- [B, 64, 1, 224, 224] #one frame of 64 channels
        x_2d = self.temporal_conv(cleaned_x)
        
        #remove time
        x_2d = x_2d.squeeze(2) 
        
        #new output- [B, 1, 32, 224, 224]
        x_2d = F.relu(self.dec_conv1(x_2d))
        
        #final output- [B, 1, 224, 224]
        mask_logits = self.final_conv(x_2d)
        
        final_mask = torch.sigmoid(mask_logits)
        
        return final_mask, attn_weights