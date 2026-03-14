import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# import torch.nn.
from model_2 import MaskGen
import random
import numpy as np
from data_utils import UCFRep_train #the custom dataset
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
    # torch.use_deterministic_algorithms(True)

seed_everything()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UCFRep_train("ucfrep_intermediate_dataset", clip_len=8, k_gap=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    model = MaskGen().to(device)
    
    num_gpus = torch.cuda.device_count() 
    print(f"Active GPUs for this session: {num_gpus}")
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    temperature = 0.07
    
    # scaler = torch.amp.GradScaler('cuda')
    model.train()
    
    for epoch in range(10):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # dataloader output- [B, T=8, C=387, H=224, W=224]
            # I need [B, C, T, H, W]. So permute:
            anchor_cpu = batch["anchor"].permute(0, 2, 1, 3, 4)
            positive_cpu = batch["positive"].permute(0, 2, 1, 3, 4)
            B = anchor_cpu.size(0)
            
            #call model for both
            # anchor_mask, _ = model(anchor)
            # pos_mask, _ = model(positive)

            anchor_mask, _ = model(anchor_cpu)
            pos_mask, _ = model(positive_cpu)

            # DINO features - slicing 768 for Base
            anchor_dino = anchor_cpu[:, :768, 0, :, :].to(device)
            pos_dino = positive_cpu[:, :768, 0, :, :].to(device)
            
            # Weighted averaging logic
            anchor_vec = torch.sum(anchor_dino * anchor_mask, dim=(2, 3)) / (torch.sum(anchor_mask, dim=(2, 3)) + 1e-8)
            pos_vec = torch.sum(pos_dino * pos_mask, dim=(2, 3)) / (torch.sum(pos_mask, dim=(2, 3)) + 1e-8)
            
            anchor_vec = F.normalize(anchor_vec, dim=1)
            pos_vec = F.normalize(pos_vec, dim=1)
            
            # InfoNCE logits
            temporal_logits = torch.matmul(anchor_vec, pos_vec.T) / temperature
            
            # Spatial Negatives
            anchor_bg_mask = 1.0 - anchor_mask
            anchor_bg_vec = F.normalize(torch.sum(anchor_dino * anchor_bg_mask, dim=(2, 3)) / (torch.sum(anchor_bg_mask, dim=(2, 3)) + 1e-8), dim=1)
            anchor_spatial_logits = torch.sum(anchor_vec * anchor_bg_vec, dim=1, keepdim=True) / temperature
            
            pos_bg_mask = 1.0 - pos_mask
            pos_bg_vec = F.normalize(torch.sum(pos_dino * pos_bg_mask, dim=(2, 3)) / (torch.sum(pos_bg_mask, dim=(2, 3)) + 1e-8), dim=1)
            pos_spatial_logits = torch.sum(anchor_vec * pos_bg_vec, dim=1, keepdim=True) / temperature

            logits = torch.cat([temporal_logits, anchor_spatial_logits, pos_spatial_logits], dim=1)
            labels = torch.arange(B).to(device)
            loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            optimizer.step()
            if batch_idx % 5 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        os.makedirs("test_models", exist_ok=True)
        checkpoint_path = os.path.join("test_models", f"mask_gen_epoch_{epoch}.pth")
        torch.save(model.module.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
if __name__ == "__main__":
    train()