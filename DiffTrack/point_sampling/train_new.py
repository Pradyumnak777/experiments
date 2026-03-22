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
    
    #using the clean hybrid dataset
    dataset = UCFRep_finetune(
        mp4_dir="UCF_Rep/train",
        pt_dir="ucfrep_intermediate_dataset"
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    
    model = DINOv2_LoRA().to(device)    
    
    num_gpus = torch.cuda.device_count() 
    print(f"Active GPUs for this session: {num_gpus}")
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    temperature = 0.07
    model.train()
    
    for epoch in range(100):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            pixels = torch.cat([batch["im1_pixels"].unsqueeze(1), batch["im2_pixels"].unsqueeze(1)], dim=1).to(device)
            flow = torch.cat([batch["im1_flow"].unsqueeze(1), batch["im2_flow"].unsqueeze(1)], dim=1).to(device)
            depth = torch.cat([batch["im1_depth"].unsqueeze(1), batch["im2_depth"].unsqueeze(1)], dim=1).to(device)
            
            b, t = pixels.shape[0], pixels.shape[1]
            
            outputs = model(pixels)
            patch_features = outputs["patch_features"]
            pred_mask = outputs["pred_mask"]
            #fix: last_attn is now a tensor [b*t, 12, 261, 261]
            last_attn = outputs["last_attn"] 

            #1. CROSS ENTROPY BRANCH
            high_res_fd_mask = get_robust_mask(flow, depth)
            high_res_fd_flat = high_res_fd_mask.view(b * t, 1, 224, 224)
            ce_target_low = F.interpolate(high_res_fd_flat, size=(16, 16), mode='nearest')
            ce_target = ce_target_low.view(b, t, 1, 16, 16)
            
            loss_ce = F.binary_cross_entropy(pred_mask, ce_target)
                        
            #2. CONTRASTIVE BRANCH
            with torch.no_grad():
                #fix: skip cls (0) and registers (1,2,3,4) to get 256 patches
                cls_attn = last_attn[:, :, 0, 5:] #[b*t, 12, 256]
                
                attn_map = cls_attn.mean(dim=1).view(b, t, 16, 16)
                thresh = torch.quantile(attn_map.view(b, t, -1), 0.9, dim=-1, keepdim=True).unsqueeze(-1)
                cr_target = (attn_map > thresh).float().unsqueeze(2)
            
            features_flat = patch_features.permute(0, 1, 3, 4, 2).reshape(-1, 768) 
            features_flat = F.normalize(features_flat, dim=1)
            cr_mask_flat = cr_target.view(-1)
            
            actor_vectors = features_flat[cr_mask_flat == 1]
            bg_vectors = features_flat[cr_mask_flat == 0]
            
            loss_cr = torch.tensor(0.0, device=device)
            
            if actor_vectors.size(0) > 1 and bg_vectors.size(0) > 0:
                sim_pos = torch.matmul(actor_vectors, actor_vectors.T) / temperature
                eye = torch.eye(sim_pos.size(0), device=device, dtype=torch.bool)
                sim_pos = sim_pos.masked_fill(eye, -1e9)
                
                sim_neg = torch.matmul(actor_vectors, bg_vectors.T) / temperature
                
                max_val = torch.max(torch.cat([sim_pos, sim_neg], dim=1), dim=1, keepdim=True)[0]
                exp_pos = torch.exp(sim_pos - max_val)
                exp_neg = torch.exp(sim_neg - max_val)
                
                prob = exp_pos.sum(dim=1) / (exp_pos.sum(dim=1) + exp_neg.sum(dim=1) + 1e-8)
                loss_cr = -torch.log(prob + 1e-8).mean()
            
            lambda_ce = 0.5
            total_loss = (lambda_ce * loss_ce) + loss_cr
            
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Total Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, CR: {loss_cr.item():.4f})")
                        
        os.makedirs("test_models", exist_ok=True)
        checkpoint_path = os.path.join("test_models", f"attn_guide_lora_dino_epoch_{epoch}.pth")
        
        #save checkpoint every 10th epoch
        if (epoch + 1) % 10 == 0:
            save_state = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save(save_state, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
if __name__ == "__main__":
    train()