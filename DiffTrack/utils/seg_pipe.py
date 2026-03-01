import os
import glob
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from torchvision.io import read_image
from torchvision.models.optical_flow import raft_large
from tqdm import tqdm
import cv2
from transformers import Sam2VideoModel, Sam2VideoProcessor
from accelerate import Accelerator

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

# def get_pca_map(img_path, dino_model):
#     #load image and get sizes
#     img = Image.open(img_path).convert('RGB')
#     w, h = img.size
    
#     transform = T.Compose([
#         T.Resize((518, 518)),
#         T.ToTensor(),
#         T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ])
#     img_tensor = transform(img).unsqueeze(0).cuda()
    
#     #get features
#     with torch.no_grad():
#         features_dict = dino_model.forward_features(img_tensor)
#         features = features_dict['x_norm_patchtokens']
        
#     #pca for main object
#     features = features.cpu().numpy()[0] 
#     pca = PCA(n_components=3)
#     pca.fit(features)
#     pca_features = pca.transform(features)
    
#     #1st component back to grid
#     patch_h, patch_w = 518 // 14, 518 // 14
#     foreground_map = pca_features[:, 0].reshape(patch_h, patch_w) 
    
#     #normalize
#     foreground_map = (foreground_map - foreground_map.min()) / (foreground_map.max() - foreground_map.min())
    
#     #resize to original
#     foreground_map = torch.tensor(foreground_map).unsqueeze(0).unsqueeze(0)
#     dino_hires = F.interpolate(foreground_map, size=(h, w), mode='bilinear').squeeze().numpy()
    
#     #heuristic check
#     if dino_hires[0,0] > 0.5:
#         dino_hires = 1 - dino_hires

#     return dino_hires

def sam_seg(fused_tensor, video_folder):
    #take in image based (loosely) on the threshold of the fused tensor, then segment
    
    #appraoch 1: video propogation segmentation (based on 1st frame only)
    device = Accelerator().device
    model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-large").to(device, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-large")
    #open concerned video
    frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    frames = [Image.open(fp).convert('RGB') for fp in frame_paths]
    
    inference_session = processor.init_video_session(
    video=frames,
    inference_device=device,
    dtype=torch.bfloat16,
    )
    
    #get input boxes from the heatmap of the first frame
    heatmap = fused_tensor[0].numpy()
    #first frame did not wokr out well..
    
    
    
    binary_region = heatmap > (np.max(heatmap) * 0.2) #already salient regions...threshold can be lowered..
    coords = np.argwhere(binary_region) #get coords
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    y_min, x_min = y_min.item(), x_min.item()
    y_max, x_max = y_max.item(), x_max.item()
    
    ann_frame_idx = 0
    obj_ids = [1] #only one box is needed..
    input_boxes = [[[x_min, y_min, x_max, y_max]]]

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=obj_ids,
        input_boxes = input_boxes
    )
    
    outputs = model(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    )
    
    video_res_masks = processor.post_process_masks(
        [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=True
    )[0]
    
    video_segments = {}
    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=True
        )[0]
        video_segments[sam2_video_output.frame_idx] = {
            obj_id: video_res_masks[i]
            for i, obj_id in enumerate(inference_session.obj_ids)
        }
    
    #now save video segments as a video
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "sam_seg_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.basename(video_folder)
    output_path = os.path.join(output_dir, f"{video_name}_sam2_tracking_overlay.mp4")
    h = int(inference_session.video_height)
    w = int(inference_session.video_width)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
    
    print("saving tracking video...")
    
    for frame_idx in sorted(video_segments.keys()):
        frame = cv2.imread(frame_paths[frame_idx])
        
        mask_tensor = video_segments[frame_idx][1] 
        mask_np = mask_tensor.squeeze().cpu().numpy()
        
        overlay = frame.copy()
        overlay[mask_np == 1] = (0, 255, 0) 
        
        blended = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        out.write(blended)
        
    out.release()
    print(f"video saved to {output_path}")
    
    return video_segments
    
    

def process_video_fusion(video_folder, name):
    #get frame paths
    frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"no frames found in {video_folder}")
        
    print(f"processing {len(frame_paths)} frames for {name}...")
    
    #load models once
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()
    raft_model = raft_large(pretrained=True, progress=False).cuda().eval()
    
    #raft transform
    raft_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
        T.Resize(size=(520, 960)),
    ])
    
    fused_scores = []
    
    #inference loop
    for i in tqdm(range(len(frame_paths) - 1), desc="fusing pca and flow"):
        img1_path = frame_paths[i]
        img2_path = frame_paths[i+1]
        
        #dino pca map
        dino_hires = get_pca_map(img1_path, dino_model)
        h, w = dino_hires.shape
        
        #raft flow
        img1_t = read_image(img1_path)
        img2_t = read_image(img2_path)
        
        batch1 = raft_transform(img1_t).unsqueeze(0).cuda()
        batch2 = raft_transform(img2_t).unsqueeze(0).cuda()
        
        with torch.no_grad():
            flow_output = raft_model(batch1, batch2)
            flow = flow_output[-1][0] 
            
        #magnitude
        flow_mag = torch.sqrt(flow[0]**2 + flow[1]**2) #this is speed..
        
        #resize flow to match dino
        flow_mag = flow_mag.unsqueeze(0).unsqueeze(0)
        
        #vectors are of different sizes..
        flow_hires = F.interpolate(flow_mag, size=(h, w), mode='bilinear').squeeze().cpu().numpy()
        
        #normalize flow
        flow_hires = (flow_hires - flow_hires.min()) / (flow_hires.max() - flow_hires.min() + 1e-6)
        
        #multiplying logic, if dino is high and flow is low, score will also be low. both should be high
        combined_score = dino_hires * flow_hires #elementwise multipliation
        fused_scores.append(torch.tensor(combined_score, dtype=torch.float32))
        
    #save sequence
    final_video_tensor = torch.stack(fused_scores, dim=0)
    output_dir = "fused_tensors"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{name}_fused_scores.pkl")
    
    with open(save_path, "wb") as f:
        pickle.dump(final_video_tensor.cpu().numpy(), f)
        
    print(f"saved fused sequence to {save_path}. shape: {final_video_tensor.shape}")
    return final_video_tensor

def save_fusion_video(frame_paths, fused_tensor, name):
    #a setup output path
    output_path = f"fused_tensors/{name}_visualization.mp4"
    
    #a get dimensions from the first frame
    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape
    
    #a setup video writer (using mp4v for compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w * 2, h))
    
    print(f"saving visualization video to {output_path}...")
    
    for i in tqdm(range(len(fused_tensor)), desc="writing video"):
        #read original frame
        frame = cv2.imread(frame_paths[i])
        
        #convert heatmap to viridis-style colors
        heatmap = (fused_tensor[i].numpy() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
        
        # side-by-side
        combined = np.hstack((frame, heatmap_color))
        
        out.write(combined)
        
    out.release()
    print("video saved successfully.")

if __name__ == "__main__":
    # run config
    video_name = "benchpress"   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_folder = os.path.join(project_root, f"videos/{video_name}")
    
    #execute
    fused_tensor = process_video_fusion(target_folder, video_name)
    
    frame_paths = sorted(glob.glob(os.path.join(target_folder, "*.jpg")))
    save_fusion_video(frame_paths, fused_tensor, video_name)
    
    
    # #performing sam_seg on this fused heatmap..
    # video_name = "swim_3"  # set your video name here
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir)
    # target_folder = os.path.join(project_root, f"videos/{video_name}")
    # fused_tensor_path = os.path.join(project_root, "fused_tensors", f"{video_name}_fused_scores.pkl")
    # with open(fused_tensor_path, "rb") as f:
    #     fused_tensor = torch.tensor(pickle.load(f))
    # sam_seg(fused_tensor, target_folder)