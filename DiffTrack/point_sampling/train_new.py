import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model_finetune import DINOv2_LoRA, get_robust_mask
import random
import numpy as np
from data_utils import UCFRep_finetune
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
    dataset = UCFRep_finetune(
        mp4_dir="UCF_Rep/train",
        pt_dir="ucfrep_intermediate_dataset", 
        k = 1
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True) #EXPERIMENT WITH BATCH NEGATIVES LATER??
    
    #init lora model with the new head
    model = DINOv2_LoRA().to(device)    
    
    num_gpus = torch.cuda.device_count() 
    print(f"Active GPUs for this session: {num_gpus}")
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    #only optimize the lora parameters AND the new seg_head
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    temperature = 0.07
    
    #can use amp safely now because rgb inputs are small
    # scaler = torch.amp.GradScaler('cuda')
    model.train()
    
    for epoch in range(100):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 1. Stack anchor and pos to create the T=2 dimension for the model
            # [B, C, H, W] -> unsqueeze -> [B, 1, C, H, W] -> concat -> [B, 2, C, H, W]
            pixels = torch.cat([batch["im1_pixels"].unsqueeze(1), batch["im2_pixels"].unsqueeze(1)], dim=1).to(device)
            flow = torch.cat([batch["im1_flow"].unsqueeze(1), batch["im2_flow"].unsqueeze(1)], dim=1).to(device)
            depth = torch.cat([batch["im1_depth"].unsqueeze(1), batch["im2_depth"].unsqueeze(1)], dim=1).to(device)
            
            b, t = pixels.shape[0], pixels.shape[1]
            

            # with torch.amp.autocast('cuda'):
            outputs = model(pixels)
            patch_features = outputs["patch_features"] # [B, 2, 768, 16, 16]
            pred_mask = outputs["pred_mask"]  # [B, 2, 1, 16, 16]

            #creating flow+depth mask
            high_res_mask = get_robust_mask(flow, depth) #[B, 2, 1, 224, 224]
            #downsample it
            high_res_mask_flat = high_res_mask.view(b * t, 1, 224, 224)
            low_res_mask = F.interpolate(high_res_mask_flat, size=(16, 16), mode='nearest')
            binary_mask = low_res_mask.view(b, t, 1, 16, 16) # [B, 2, 1, 16, 16]..but will downsampling ruin the mask??
            
            #cross entropy loss [CHECK PAPER: 'wholly unsupervised!..']
            loss_ce = F.binary_cross_entropy(pred_mask, binary_mask)
            
            #contrastive loss
            features_flat = patch_features.permute(0, 1, 3, 4, 2).reshape(-1, 768) # [num patches, 768]
            features_flat = F.normalize(features_flat, dim=1)
            
            mask_flat = binary_mask.view(-1)
            
            #seperation
            actor_vectors = features_flat[mask_flat == 1]
            bg_vectors = features_flat[mask_flat == 0]
            
            loss_cr = torch.tensor(0.0, device=device)
            
            if actor_vectors.size(0) > 1 and bg_vectors.size(0) > 0:
                
                #similarity between all actors (positive set!)
                sim_pos = torch.matmul(actor_vectors, actor_vectors.T) / temperature
                
                # mask out the diagonal (self similarity avoiding)
                eye = torch.eye(sim_pos.size(0), device=device, dtype=torch.bool)
                sim_pos = sim_pos.masked_fill(eye, -1e9)
                
                # imilarity between actors and background (negative set)
                sim_neg = torch.matmul(actor_vectors, bg_vectors.T) / temperature
                
                #math
                max_val = torch.max(torch.cat([sim_pos, sim_neg], dim=1), dim=1, keepdim=True)[0]
                exp_pos = torch.exp(sim_pos - max_val)
                exp_neg = torch.exp(sim_neg - max_val)
                
                #InfoNCE
                sum_exp_pos = exp_pos.sum(dim=1)
                sum_exp_neg = exp_neg.sum(dim=1)
                prob = sum_exp_pos / (sum_exp_pos + sum_exp_neg + 1e-8)
                
                # final CR loss is the negative log of that probability
                loss_cr = -torch.log(prob + 1e-8).mean()
            
            total_loss = loss_ce + loss_cr
                
            # #scale and step
            # scaler.scale(total_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Total Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, CR: {loss_cr.item():.4f})")
        
        os.makedirs("test_models", exist_ok=True)
        checkpoint_path = os.path.join("test_models", f"new3d_lora_dino_epoch_{epoch}.pth")
        # Save checkpoint every 50th epoch (50, 100, ...)
        if (epoch + 1) % 25 == 0:
            # safeguard for DataParallel saving
            save_state = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save(save_state, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
if __name__ == "__main__":
    train()