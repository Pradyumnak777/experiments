import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from model_finetune import DINOv2_LoRA
import matplotlib.pyplot as plt
from collections import OrderedDict
from transformers import AutoModel

checkpoint_path = "test_models/attn_guide_lora_dino_epoch_9.pth" 
# video_path = "vids_mp4/swim_2.mp4"
video_path = "UCF_Rep/val/v_Biking_g24_c06.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize():
    model = DINOv2_LoRA().to(device)
    
    #load weights and strip the module prefix from dataparallel
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"loaded weights from {checkpoint_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    raw_frames = []
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    #grabbing the first 2 frames to satisfy the model's T=2 expectation
    for _ in range(2): 
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(cv2.resize(frame_rgb, (224, 224)))
        
        t_frame = torch.from_numpy(raw_frames[-1]).permute(2, 0, 1).float() / 255.0
        t_frame = t_frame.to(device) 
        t_frame = (t_frame - norm_mean) / norm_std
        frames.append(t_frame)
    cap.release()

    input_tensor = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_mask = outputs["pred_mask"] #student prediction [1, 2, 1, 16, 16]
        last_attn = outputs["last_attn"]    #teacher attention tensor [2, 12, 261, 261]

    #plotting a comparison: original vs. teacher vs. student
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    #the raw input
    axes[0].imshow(raw_frames[0])
    axes[0].set_title("original frame")
    axes[0].axis('off')

    # grab last layer, batch 0, cls token (idx 0), and all 256 patches (1:)
    #we want batch 0 (frame 0), all 12 heads, the cls token (idx 0), and 256 patches (5:)
    cls_attn_heads = last_attn[0, :, 0, 5:] #[12, 256]
    
    #average across the 12 heads to get one single saliency map
    # shape becomes [256]
    cls_attn_mean = cls_attn_heads.mean(dim=0)
    
    # now reshape the 256 patches into a 16x16 grid
    # shape becomes [1, 1, 16, 16] for interpolate
    cls_attn_grid = cls_attn_mean.view(1, 1, 16, 16)
    
    #interpolation
    teacher_map = F.interpolate(cls_attn_grid, size=(224, 224), mode='bilinear').squeeze().cpu().numpy()
    
    t_min, t_max = teacher_map.min(), teacher_map.max()
    teacher_map = (teacher_map - t_min) / (t_max - t_min + 1e-8)
    
    axes[1].imshow(raw_frames[0])
    axes[1].imshow(teacher_map, cmap='jet', alpha=0.5)
    axes[1].set_title("teacher ([cls] attention)")
    axes[1].axis('off')

    #3. the student (your seg head)
    pred_map = F.interpolate(predicted_mask[0, 0].unsqueeze(0), size=(224, 224), mode='bilinear').squeeze().cpu().numpy()
    
    axes[2].imshow(raw_frames[0])
    axes[2].imshow(pred_map, cmap='jet', alpha=0.5)
    axes[2].set_title("student (seg head)")
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs("point_sampling/finetuned_test_new/", exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_name = f"point_sampling/finetuned_test_new/{video_name}_comparison.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"saved comparison to {save_name}")


def visualize_raw_teacher():
    #load just the base model with registers directly from huggingface
    base_model_name = 'facebook/dinov2-with-registers-base'
    print(f"loading raw base model: {base_model_name}")
    
    #force output_attentions=True right at the source
    teacher_model = AutoModel.from_pretrained(base_model_name, output_attentions=True).to(device)
    teacher_model.eval()

    cap = cv2.VideoCapture(video_path)
    #we only need 1 frame since we aren't passing it to your T=2 model
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("failed to read video")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(frame_rgb, (224, 224))
    
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    t_frame = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0
    t_frame = t_frame.to(device) 
    t_frame = (t_frame - norm_mean) / norm_std
    
    #add batch dimension [1, 3, 224, 224]
    input_tensor = t_frame.unsqueeze(0)

    with torch.no_grad():
        outputs = teacher_model(input_tensor)
        attentions = outputs.attentions 

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    #1. original
    axes[0].imshow(raw_img)
    axes[0].set_title("original frame")
    axes[0].axis('off')

    #2. teacher attention
    #attentions[-1] shape is [1, 12, 261, 261]
    #skip 1 cls + 4 registers -> start at index 5
    cls_attn_heads = attentions[-1][0, :, 0, 5:] #[12, 256]
    cls_attn_mean = cls_attn_heads.mean(dim=0) #[256]
    cls_attn_grid = cls_attn_mean.view(1, 1, 16, 16)
    
    #upsample and normalize
    teacher_map = F.interpolate(cls_attn_grid, size=(224, 224), mode='bilinear').squeeze().cpu().numpy()
    
    t_min, t_max = teacher_map.min(), teacher_map.max()
    teacher_map = (teacher_map - t_min) / (t_max - t_min + 1e-8)
    
    axes[1].imshow(raw_img)
    axes[1].imshow(teacher_map, cmap='jet', alpha=0.5)
    axes[1].set_title("raw dinov2-registers teacher")
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs("point_sampling/finetuned_test_new/", exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_name = f"point_sampling/finetuned_test_new/{video_name}_raw_teacher.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"saved raw teacher to {save_name}")

if __name__ == "__main__":
    visualize()
    # visualize_raw_teacher()