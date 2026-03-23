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
            
            #data_utils now returns pre-stacked sequences [b, 3, c, h, w]
            pixels = batch["pixels"].to(device)
            flow = batch["flow"].to(device)
            # depth = batch["depth"].to(device)
            
            b, t = pixels.shape[0], pixels.shape[1]
            
            outputs = model(pixels)
            patch_features = outputs["patch_features"]
            pred_mask = outputs["pred_mask"]

            #1- FLOW MASK!
            high_res_fd_mask = get_robust_mask(flow) #[b, t, 1, 224, 224]
            high_res_fd_flat = high_res_fd_mask.view(b * t, 1, 224, 224)
            
            #downsample to 16x16 for the loss functions
            target_low = F.interpolate(high_res_fd_flat, size=(16, 16), mode='nearest')
            target_low = target_low.view(b, t, 1, 16, 16)
            
            #2- CROSS ENTROPY
            #teaches the seg_head to find the moving actor
            loss_ce = F.binary_cross_entropy(pred_mask, target_low)
                        
            #3- USING SAME FLOW MASK TO DO CONTRASTIVE LEARNING TOO
            #teaches the lora weights to temporally group those actor features
            features_flat = patch_features.permute(0, 1, 3, 4, 2).reshape(-1, 768)  #b*t, 768, 16, 16
            features_flat = F.normalize(features_flat, dim=1)
            cr_mask_flat = target_low.view(-1)
            
            actor_vectors = features_flat[cr_mask_flat == 1] #actor patches where flow_mask gives high values
            bg_vectors = features_flat[cr_mask_flat == 0]#bg patches where flow mask has low value..
            
            loss_cr = torch.tensor(0.0, device=device)
            
            if actor_vectors.size(0) > 1 and bg_vectors.size(0) > 0:
                #this similarity matrix now naturally spans across the t=3 chunk
                sim_pos = torch.matmul(actor_vectors, actor_vectors.T) / temperature
                eye = torch.eye(sim_pos.size(0), device=device, dtype=torch.bool)
                sim_pos = sim_pos.masked_fill(eye, -1e9)
                
                sim_neg = torch.matmul(actor_vectors, bg_vectors.T) / temperature
                
                max_val = torch.max(torch.cat([sim_pos, sim_neg], dim=1), dim=1, keepdim=True)[0]
                exp_pos = torch.exp(sim_pos - max_val)
                exp_neg = torch.exp(sim_neg - max_val)
                
                prob = exp_pos.sum(dim=1) / (exp_pos.sum(dim=1) + exp_neg.sum(dim=1) + 1e-8)
                loss_cr = -torch.log(prob + 1e-8).mean()
            
            lambda_ce = 1.0 
            total_loss = (lambda_ce * loss_ce) + loss_cr
            
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Total Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, CR: {loss_cr.item():.4f})")
            
            if batch_idx % 50 == 0:
                save_debug_image(pixels[0, 1], high_res_fd_mask[0, 1], pred_mask[0, 1], epoch, batch_idx)
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Total: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, CR: {loss_cr.item():.4f})")
                        
        os.makedirs("test_models", exist_ok=True)
        checkpoint_path = os.path.join("test_models", f"physics_guide_lora_dino_epoch_{epoch}.pth")
        
        #save checkpoint every 10th epoch
        if (epoch + 1) % 10 == 0:
            save_state = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save(save_state, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
        
def save_debug_image(img_tensor, gt_mask, pred_mask, epoch, batch_idx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs("train_debug", exist_ok=True)
    
    # Use .detach() before .cpu().numpy()
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    gt = gt_mask.detach().cpu().squeeze().numpy()
    
    # Detach here so F.interpolate doesn't try to track gradients
    pred_upsampled = F.interpolate(pred_mask.detach().unsqueeze(0), size=(224, 224), mode='bilinear')
    pred = pred_upsampled.cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title("Input")
    axes[1].imshow(gt, cmap='jet'); axes[1].set_title("Teacher Mask")
    axes[2].imshow(pred, cmap='jet'); axes[2].set_title("Student Pred")
    
    plt.savefig(f"train_debug/epoch{epoch}_batch{batch_idx}.png")
    plt.close(fig) # Explicitly close fig to save memory
    
        
if __name__ == "__main__":
    train()