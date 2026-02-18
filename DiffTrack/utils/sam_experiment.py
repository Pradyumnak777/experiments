import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import pipeline
from PIL import Image

#setting up sam2

def sam2_autoseg(image_path):
    output_dir = 'sam2_masks_experiment'
    checkpoint = "facebook/sam2.1-hiera-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_generator = pipeline(model=checkpoint, task="mask-generation", device=device)
    raw_image = Image.open(image_path).convert("RGB")
    outputs = mask_generator(
    raw_image,
    points_per_side=16,   
    points_per_batch=64, # Process the 256 points in 4 batches of 64
    pred_iou_thresh=0.75,     
    stability_score_thresh=0.8 
    )
    
    #simulating 'min_mask_region_area' argument from sam1 in sam2
    filtered_masks = []
    for mask_tensor, score in zip(outputs["masks"], outputs["scores"]):
        # Calculate Area
        area = mask_tensor.sum().item()
        
        # Apply the "min_mask_region_area=800" logic manually
        if area >= 800:
            filtered_masks.append({
                "segmentation": mask_tensor.cpu().numpy(),
                "area": area,
                "score": score.item()
            })
    masks = sorted(filtered_masks, key=lambda x: x['area'], reverse=True)[:4]
    
    
    # masks = outputs["masks"].squeeze(0)
    # areas = masks.sum(dim=(1, 2))
    # top_indices = torch.argsort(areas, descending=True)[:4]
    # final_masks = masks[top_indices]
    # final_masks = final_masks.cpu().numpy()
    # masks = sorted(masks, key=(lambda x: x['area']), reverse=True)[:4]#only keep the 4 biggest masks
    
    return raw_image, masks, output_dir
    
def sam1_autoseg(image_path):
    #setup paths
    checkpoint_path = "/scratch/pbk5339/thesis/DiffTrack/sam_vit_h_4b8939.pth"
    output_dir = 'sam_masks_experiment'
    # image_path = '/scratch/pbk5339/thesis/DiffTrack/videos/swim/frames_001.jpg'

    os.makedirs(output_dir, exist_ok=True)

    #load the model onto gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)

    #init the generator with strict settings to keep it clean
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16, #fineness of the mask..? 256x256 grid...
        pred_iou_thresh=0.75,#confidence score...putting it high
        stability_score_thresh=0.8, #stability(?)
        min_mask_region_area=800 #size below 1000 pixels..(32x32?) ommited..
    )

    #load image and run sam
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    #only keep the 4 biggest masks
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)[:4]
    return image, masks, output_dir

#function to overlay masks and save
def save_anns(image, anns, output_path):
    image = np.array(image)
    if len(anns) == 0: return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    #create the colorful overlay (color doesnt mean smae object!!)
    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.4]])
        img[m] = color_mask
    
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    image_path = '/scratch/pbk5339/thesis/DiffTrack/videos/swim/frames_001.jpg'
    # image, masks, output_dir = sam1_autoseg(image_path)
    image, masks, output_dir = sam2_autoseg(image_path)


    #naming logic
    parts = os.path.normpath(image_path).split(os.sep)
    if "videos" in parts:
        videos_idx = parts.index("videos")
        video_name = parts[videos_idx + 1] if videos_idx + 1 < len(parts) else "unknown"
    else:
        video_name = "unknown"
    frame_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{video_name}_{frame_name}_masked.png")
    os.makedirs(output_dir, exist_ok=True)
    save_anns(image, masks, save_path)
    print(f"Successfully saved {len(masks)} masks to {save_path}")
# #save the final result
# save_path = os.path.join(output_dir, 'swim_frame_1_masked.png')
# save_anns(image, masks, save_path)
# print(f"Successfully saved {len(masks)} masks to {save_path}")