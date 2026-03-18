import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model_finetune import DINOv2_LoRA, get_robust_mask
import random
import numpy as np
from data_utils import UCFRep_Finetune
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #use the new hybrid dataset
    dataset = UCFRep_Finetune(
        mp4_dir="UCF_Rep/train", 
        pt_dir="ucfrep_intermediate_dataset", 
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True) #EXPERIMENT WITH BATCH NEGATIVES LATER??
    
    
    '''
    now make the mask using flow + depth:
    '''
    
    
    
    #init lora model
    model = DINOv2_LoRA().to(device)
    
    num_gpus = torch.cuda.device_count() 
    print(f"Active GPUs for this session: {num_gpus}")
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    #only optimize the lora parameters, not the frozen backbone
    #learning rate slightly higher for lora is standard practice
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    temperature = 0.07
    
    #we can use amp safely now because rgb inputs are small
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    
    for epoch in range(10):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            #1. move rgb frames to gpu for the model
            anchor_pixels = batch["anchor_pixels"].to(device) #[b, t, 3, 224, 224]
            pos_pixels = batch["pos_pixels"].to(device)
            B, T = anchor_pixels.shape[0], anchor_pixels.shape[1]
            
            #2. keep physics on cpu temporarily to save vram
            #they are only needed for the loss math later
            anchor_flow = batch["anchor_flow"]
            anchor_depth = batch["anchor_depth"]
            pos_flow = batch["pos_flow"]
            pos_depth = batch["pos_depth"]

            with torch.amp.autocast('cuda'):
                #dino features out - [b, t, 768, 16, 16]
                anchor_dino = model(anchor_pixels)
                pos_dino = model(pos_pixels)

                #calculate masks using robust physics heuristic
                anchor_mask_224 = get_robust_mask(anchor_flow, anchor_depth).to(device) #[b, t, 1, 224, 224]
                pos_mask_224 = get_robust_mask(pos_flow, pos_depth).to(device)
                
                curr_bt = anchor_mask_224.shape[0] * anchor_mask_224.shape[1]
                
                #downsample masks from 224x224 to 16x16 to match dino patches
                #we reshape to 4d for interpolate, then back to 5d
                anchor_mask = F.interpolate(
                    anchor_mask_224.view(curr_bt, 1, 224, 224), 
                    size=(16, 16), 
                    mode='nearest'
                ).view(B, T, 1, 16, 16)
                
                pos_mask = F.interpolate(
                    pos_mask_224.view(curr_bt, 1, 224, 224), 
                    size=(16, 16), 
                    mode='nearest'
                ).view(B, T, 1, 16, 16)

                #weighted averaging
                #we sum across time (dim=1) and space (dim=3,4) to get one single vector per video
                anchor_vec = torch.sum(anchor_dino * anchor_mask, dim=(1, 3, 4)) / (torch.sum(anchor_mask, dim=(1, 3, 4)) + 1e-8)
                pos_vec = torch.sum(pos_dino * pos_mask, dim=(1, 3, 4)) / (torch.sum(pos_mask, dim=(1, 3, 4)) + 1e-8)
                
                anchor_vec = F.normalize(anchor_vec, dim=1) #[b, 768]
                pos_vec = F.normalize(pos_vec, dim=1) #[b, 768]
                
                '''
                contrastive learning loss! - infoNCE
                '''
                #infonce logits (temporal positives/negatives)
                temporal_logits = torch.matmul(anchor_vec, pos_vec.T) / temperature #diagonals should tend to 1(positives)
                
                #spatial negatives (background of the same videos)
                anchor_bg_mask = 1.0 - anchor_mask
                anchor_bg_vec = F.normalize(torch.sum(anchor_dino * anchor_bg_mask, dim=(1, 3, 4)) / (torch.sum(anchor_bg_mask, dim=(1, 3, 4)) + 1e-8), dim=1)
                anchor_spatial_logits = torch.sum(anchor_vec * anchor_bg_vec, dim=1, keepdim=True) / temperature
                
                pos_bg_mask = 1.0 - pos_mask
                pos_bg_vec = F.normalize(torch.sum(pos_dino * pos_bg_mask, dim=(1, 3, 4)) / (torch.sum(pos_bg_mask, dim=(1, 3, 4)) + 1e-8), dim=1)
                pos_spatial_logits = torch.sum(anchor_vec * pos_bg_vec, dim=1, keepdim=True) / temperature

                #dot prod b/w bg_vec and vec s hould be very low..
                
                #combine and calculate loss
                logits = torch.cat([temporal_logits, anchor_spatial_logits, pos_spatial_logits], dim=1)
                labels = torch.arange(B).to(device)
                loss = F.cross_entropy(logits, labels)
            
            #scale and step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 5 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        os.makedirs("test_models", exist_ok=True)
        checkpoint_path = os.path.join("test_models", f"lora_dino_epoch_{epoch}.pth")
        
        #safeguard for data parallel saving
        save_state = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(save_state, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
if __name__ == "__main__":
    train()