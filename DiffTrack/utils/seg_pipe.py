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
from dino_exp import maskcut_method
from transformers import AutoModel
# import torch.multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor
# import queue

# #for parallel stuff
# worker_dino = None
# worker_raft = None
# worker_device = None
# worker_transform = None

# #ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

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

def sam_seg(fused_tensor, video_folder, mask = None):
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
    
    
    
    # binary_region = heatmap > (np.max(heatmap) * 0.2) #already salient regions...threshold can be lowered..\
    binary_region = heatmap > 0.45
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
    if mask is None:
        output_path = os.path.join(output_dir, f"{video_name}_sam2_tracking_overlay.mp4")
    else:
        output_path = os.path.join(output_dir, f"{video_name}_{mask}_sam2_tracking_overlay.mp4")
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
    

def get_camera_flow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    points_img1 = cv2.goodFeaturesToTrack(gray1, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    
    #track these into the 2nd frame
    points_img2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, points_img1, None) #this is the lucas kanade method
    
    valid_points_img1 = points_img1[status == 1]
    valid_points_img2 = points_img2[status == 1]
    
    #getting homography estimate (RANSAC)..
    H, mask = cv2.findHomography(valid_points_img1, valid_points_img2, cv2.RANSAC, 3.0)
    
    if H is None:
        #no answer!
        return np.zeros((gray1.shape[0], gray1.shape[1], 2), dtype=np.float32)
    
    #make grid
    h, w = gray1.shape
    y, x = np.mgrid[0:h, 0:w]
    coords = np.stack((x, y), axis=-1).astype(np.float32)
    
    #apply homography on frame1's grid
    coords_flat = coords.reshape(-1, 1, 2)
    coords_warped_flat = cv2.perspectiveTransform(coords_flat, H)
    coords_warped = coords_warped_flat.reshape(h, w, 2)
    
    #final
    camera_flow = coords_warped - coords
    
    return camera_flow


def process_video_fusion(video_folder, name):
    #get frame paths
    frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"no frames found in {video_folder}")
        
    print(f"processing {len(frame_paths)} frames for {name}...")
    
    #load models once
    model_name = 'facebook/dinov2-small'
    dino_model = AutoModel.from_pretrained(model_name, output_attentions=True).cuda()
    dino_model.eval()

    raft_model = raft_large(pretrained=True, progress=False).cuda().eval()
    
    #raft transform
    raft_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
        T.Resize(size=(520, 960)),
    ])
    
    fused_scores_mask1 = []
    fused_scores_mask2 = []
    
    
    #inference loop
    for i in tqdm(range(len(frame_paths) - 1), desc="fusing pca and flow"):
        img1_path = frame_paths[i]
        img2_path = frame_paths[i+1]
        
        #now, cutLER based on Dinov2 will give out 2 masks, for 2 salient objects
        
        masks = maskcut_method(img1_path, num_objects = 2, dino_model = dino_model)
        
        mask1 = masks[0]
        mask2 = masks[1]
        h, w = mask1.shape
        
        #raft flow
        img1_t = read_image(img1_path)
        img2_t = read_image(img2_path)
        
        batch1 = raft_transform(img1_t).unsqueeze(0).cuda()
        batch2 = raft_transform(img2_t).unsqueeze(0).cuda()
        
        with torch.no_grad():
            flow_output = raft_model(batch1, batch2)
            flow = flow_output[-1][0] 
            
        #camera motion cancellation (method 1: classical)
        img1_np = cv2.imread(img1_path) 
        img2_np = cv2.imread(img2_path)
        
        camera_flow_np = get_camera_flow(img1_np, img2_np) #shape is (h, w, 2)
        camera_flow = torch.tensor(camera_flow_np).permute(2, 0, 1).cuda()
        
        camera_flow_resized = F.interpolate(
            camera_flow.unsqueeze(0), 
            size=(flow.shape[1], flow.shape[2]), 
            mode='bilinear'
        ).squeeze(0)
        
        corrected_flow = flow - camera_flow_resized
        
        #magnitude
        flow_mag = torch.sqrt(corrected_flow[0]**2 + corrected_flow[1]**2)
        
        #resize flow to match mask dimensions
        flow_mag = flow_mag.unsqueeze(0).unsqueeze(0)
        flow_hires = F.interpolate(flow_mag, size=(h, w), mode='bilinear').squeeze().cpu().numpy()
        
        #normalize flow
        flow_hires = (flow_hires - flow_hires.min()) / (flow_hires.max() - flow_hires.min() + 1e-6)
        
        #compute fused scores for both masks
        combined_score_mask1 = mask1 * flow_hires
        combined_score_mask2 = mask2 * flow_hires
        
        fused_scores_mask1.append(torch.tensor(combined_score_mask1, dtype=torch.float32))
        fused_scores_mask2.append(torch.tensor(combined_score_mask2, dtype=torch.float32))
        
    #save sequences for both masks
    output_dir = "fused_tensors"
    os.makedirs(output_dir, exist_ok=True)
    
    final_video_tensor_mask1 = torch.stack(fused_scores_mask1, dim=0)
    save_path_mask1 = os.path.join(output_dir, f"{name}_mask1_fused_scores.pkl")
    with open(save_path_mask1, "wb") as f:
        pickle.dump(final_video_tensor_mask1.cpu().numpy(), f)
    print(f"saved mask1 fused sequence to {save_path_mask1}. shape: {final_video_tensor_mask1.shape}")
    
    final_video_tensor_mask2 = torch.stack(fused_scores_mask2, dim=0)
    save_path_mask2 = os.path.join(output_dir, f"{name}_mask2_fused_scores.pkl")
    with open(save_path_mask2, "wb") as f:
        pickle.dump(final_video_tensor_mask2.cpu().numpy(), f)
    print(f"saved mask2 fused sequence to {save_path_mask2}. shape: {final_video_tensor_mask2.shape}")
    
    return final_video_tensor_mask1, final_video_tensor_mask2

def visualize_fused_frame(frame_path, fused_tensor_value, output_path=None):
    #read original frame
    frame = cv2.imread(frame_path)
    h, w, _ = frame.shape
    
    #convert heatmap to viridis-style colors
    heatmap = (fused_tensor_value * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    
    #side-by-side
    combined = np.hstack((frame, heatmap_color))
    
    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"saved frame visualization to {output_path}")
    
    return combined

def save_fusion_video(frame_paths, fused_tensor, name):
    #setup output path
    output_path = f"fused_tensors/{name}_visualization.mp4"
    
    #a get dimensions from the first frame
    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape
    
    #setup video writer (using mp4v for compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w * 2, h))
    
    print(f"saving visualization video to {output_path}...")
    
    for i in tqdm(range(len(fused_tensor)), desc="writing video"):
        combined = visualize_fused_frame(frame_paths[i], fused_tensor[i].numpy())
        out.write(combined)
        
    out.release()
    print("video saved successfully.")


if __name__ == "__main__":
    # # run config
    # video_name = "swim_2"   
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir)
    # target_folder = os.path.join(project_root, f"videos/{video_name}")
    
    # #execute
    # fused_tensor_mask1, fused_tensor_mask2 = process_video_fusion(target_folder, video_name)
    
    # frame_paths = sorted(glob.glob(os.path.join(target_folder, "*.jpg")))
    # save_fusion_video(frame_paths, fused_tensor_mask1, f"{video_name}_mask1")
    # save_fusion_video(frame_paths, fused_tensor_mask2, f"{video_name}_mask2")
    
    
    # #performing sam_seg on this fused heatmap..
    # video_name = "swim"  # set your video name here
    # mask_result_name = "swim_mask1"
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir)
    # target_folder = os.path.join(project_root, f"videos/{video_name}")
    # fused_tensor_path = os.path.join(project_root, "fused_tensors", f"{mask_result_name}_fused_scores.pkl")
    # with open(fused_tensor_path, "rb") as f:
    #     fused_tensor = torch.tensor(pickle.load(f))
    # sam_seg(fused_tensor, target_folder, mask = mask_result_name)
    
    # visualize specific frame from fused tensor
    video_name = "swim_2"  
    mask_result_name = "swim_2_mask2"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_folder = os.path.join(project_root, f"videos/{video_name}")
    
    # load fused tensor from pkl
    fused_tensor_path = os.path.join(project_root, "fused_tensors", f"{mask_result_name}_fused_scores.pkl")
    with open(fused_tensor_path, "rb") as f:
        fused_tensor = torch.tensor(pickle.load(f))
    
    # load frame paths
    frame_paths = sorted(glob.glob(os.path.join(target_folder, "*.jpg")))
    
    output_dir = os.path.join(project_root, "fused_frame_visualization")
    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0  # change this to visualize different frames
    output_frame_path = os.path.join(output_dir, f"{mask_result_name}_frame_{frame_idx}.png")
    visualize_fused_frame(frame_paths[frame_idx], fused_tensor[frame_idx].numpy(), output_path=output_frame_path)