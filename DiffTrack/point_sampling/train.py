import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MaskGen, RepMask
import os
from data_utils import UCFRep_train #the custom dataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UCFRep_train("ucfrep_intermediate_dataset", clip_len=8, k_gap=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    model = MaskGen().to(device)
    model = MaskGen().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    temperature = 0.07
    model.train()
    for epoch in range(10):
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # dataloader output- [B, T=8, C=387, H=224, W=224]
            # I need [B, C, T, H, W]. So permute:
            anchor = batch["anchor"].permute(0, 2, 1, 3, 4).to(device)
            positive = batch["positive"].permute(0, 2, 1, 3, 4).to(device)
            B = anchor.size(0)
            
            #call model for both
            anchor_mask, _ = model(anchor)
            with torch.no_grad():
                m_mean = anchor_mask.mean().item()
                m_max = anchor_mask.max().item()
                m_min = anchor_mask.min().item()
                
                # Check if the mask is actually doing something
                print(f"--- [Batch {batch_idx}] Mask Health: Mean={m_mean:.4f}, Max={m_max:.4f}, Min={m_min:.4f} ---")
            pos_mask, _ = model(positive)
            
    
            #dino features needed for infoNCE loss (it is the main comparison)
            anchor_dino = anchor[:, :384, 0, :, :] # [B, 384, 224, 224]
            pos_dino = positive[:, :384, 0, :, :]  # [B, 384, 224, 224]
            
            #1D vector is needed [B, 384]..multiplying with generated mask and averaging (IS THIS EFFECTIVE??)
            #add 1e-8 to avoid dividing by zero if the mask is empty
            anchor_vec = torch.sum(anchor_dino * anchor_mask, dim=(2, 3)) / (torch.sum(anchor_mask, dim=(2, 3)) + 1e-8)
            pos_vec = torch.sum(pos_dino * pos_mask, dim=(2, 3)) / (torch.sum(pos_mask, dim=(2, 3)) + 1e-8)
            
            #normalize
            anchor_vec = F.normalize(anchor_vec, dim=1)
            pos_vec = F.normalize(pos_vec, dim=1)
            
            '''
            infoNCE loss step
            1. TEMPORAL negs- between different batches (A2, P2, A3,..; if current is A1-P1, look at kindle for clarity)
            2. SPATIAL negs- backgorund mask of same image + background of positive image
            '''
            temporal_logits = torch.matmul(anchor_vec, pos_vec.T) / temperature
            
            #for anchor background
            anchor_bg_mask = 1.0 - anchor_mask
            anchor_bg_vec = torch.sum(anchor_dino * anchor_bg_mask, dim=(2, 3)) / (torch.sum(anchor_bg_mask, dim=(2, 3)) + 1e-8)
            anchor_bg_vec = F.normalize(anchor_bg_vec, dim=1)
            anchor_spatial_logits = torch.sum(anchor_vec * anchor_bg_vec, dim=1, keepdim=True) / temperature # [B, 1]
            
            #for positive backgoirund
            pos_bg_mask = 1.0 - pos_mask
            pos_bg_vec = torch.sum(pos_dino * pos_bg_mask, dim=(2, 3)) / (torch.sum(pos_bg_mask, dim=(2, 3)) + 1e-8)
            pos_bg_vec = F.normalize(pos_bg_vec, dim=1)
            pos_spatial_logits = torch.sum(anchor_vec * pos_bg_vec, dim=1, keepdim=True) / temperature # [B, 1]

            
            #combine them 
            logits = torch.cat([temporal_logits, anchor_spatial_logits, pos_spatial_logits], dim=1)
            
            labels = torch.arange(B).to(device)
            
            #pushes the diagonal to 1 and the off-diagonals (negatives) to 0
            loss = F.cross_entropy(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        os.makedirs("test_models", exist_ok=True)
        checkpoint_path = os.path.join("test_models", f"mask_gen_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
if __name__ == "__main__":
    train()