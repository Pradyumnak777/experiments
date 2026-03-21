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
        co_seg_dir="co-segmentation"
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
            
            # grabbing the co-segmentation masks for contrastive loss
            # these come in as [B, 37, 37]
            mask1_coseg = batch["mask1"].to(device)
            mask2_coseg = batch["mask2"].to(device)
            
            b, t = pixels.shape[0], pixels.shape[1]
            

            outputs = model(pixels)
            patch_features = outputs["patch_features"] # [B, 2, 768, 16, 16]
            pred_mask = outputs["pred_mask"]  # [B, 2, 1, 16, 16]

            # 1. CROSS ENTROPY BRANCH (using flow + depth)
            # this tells the model 'what' is the actor
            high_res_fd_mask = get_robust_mask(flow, depth) # [B, 2, 1, 224, 224]
            high_res_fd_flat = high_res_fd_mask.view(b * t, 1, 224, 224)
            ce_target_low = F.interpolate(high_res_fd_flat, size=(16, 16), mode='nearest')
            ce_target = ce_target_low.view(b, t, 1, 16, 16)
            
            loss_ce = F.binary_cross_entropy(pred_mask, ce_target)
                        
            # 2. CONTRASTIVE BRANCH (using co-segmentation masks)
            # this tells the model 'how' to group features cleanly
            
            # first, interpolate the 37x37 coseg masks to 16x16 patch grid
            coseg_stack = torch.stack([mask1_coseg, mask2_coseg], dim=1) # [B, 2, 37, 37]
            coseg_stack = coseg_stack.unsqueeze(2) # [B, 2, 1, 37, 37]
            
            cr_target_flat = F.interpolate(coseg_stack.view(b*t, 1, 37, 37), size=(16, 16), mode='nearest')
            cr_target = cr_target_flat.view(b, t, 1, 16, 16)
            
            # flatten features and masks for contrastive math
            features_flat = patch_features.permute(0, 1, 3, 4, 2).reshape(-1, 768) 
            features_flat = F.normalize(features_flat, dim=1)
            cr_mask_flat = cr_target.view(-1)
            
            actor_vectors = features_flat[cr_mask_flat == 1]
            bg_vectors = features_flat[cr_mask_flat == 0]
            
            loss_cr = torch.tensor(0.0, device=device)
 
            
            if actor_vectors.size(0) > 1 and bg_vectors.size(0) > 0:
                # similarity between all actor patches across the whole batch
                sim_pos = torch.matmul(actor_vectors, actor_vectors.T) / temperature
                eye = torch.eye(sim_pos.size(0), device=device, dtype=torch.bool)
                sim_pos = sim_pos.masked_fill(eye, -1e9)
                
                # similarity between actors and background
                sim_neg = torch.matmul(actor_vectors, bg_vectors.T) / temperature
                
                # logsumexp trick for stability
                max_val = torch.max(torch.cat([sim_pos, sim_neg], dim=1), dim=1, keepdim=True)[0]
                exp_pos = torch.exp(sim_pos - max_val)
                exp_neg = torch.exp(sim_neg - max_val)
                
                prob = exp_pos.sum(dim=1) / (exp_pos.sum(dim=1) + exp_neg.sum(dim=1) + 1e-8)
                loss_cr = -torch.log(prob + 1e-8).mean()
            
            total_loss = loss_ce + (0.5 * loss_cr)
                
            # #scale and step
            # scaler.scale(total_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Total Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, CR: {loss_cr.item():.4f})")
        
        os.makedirs("test_models", exist_ok=True)
        checkpoint_path = os.path.join("test_models", f"newv2_lora_dino_epoch_{epoch}.pth")
        # Save checkpoint every 50th epoch (50, 100, ...)
        if (epoch + 1) % 10 == 0:
            # safeguard for DataParallel saving
            save_state = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save(save_state, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
if __name__ == "__main__":
    train()