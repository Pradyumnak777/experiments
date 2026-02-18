import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

#setup paths
checkpoint_path = "/scratch/pbk5339/thesis/DiffTrack/sam_vit_h_4b8939.pth"
output_dir = 'sam_masks_experiment'
image_path = '/scratch/pbk5339/thesis/DiffTrack/videos/swim/frames_001.jpg'

os.makedirs(output_dir, exist_ok=True)

#load the model onto gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
sam.to(device=device)

#init the generator with strict settings to keep it clean
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.71,#confidence score...putting it high
    stability_score_thresh=0.92, #stability(?)
    min_mask_region_area=1000 #size below 1000 pixels..(32x32?) ommited..
)

#load image and run sam
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)

#only keep the 4 biggest masks
masks = sorted(masks, key=(lambda x: x['area']), reverse=True)[:4]

#function to overlay masks and save
def save_anns(image, anns, output_path):
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

#save the final result
save_path = os.path.join(output_dir, 'frame_0005_masked.png')
save_anns(image, masks, save_path)
print(f"Successfully saved {len(masks)} masks to {save_path}")