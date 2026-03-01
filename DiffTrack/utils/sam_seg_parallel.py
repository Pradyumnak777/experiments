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
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import queue
import pynvml

#for parallel stuff
def get_free_gpus(limit=2):
    pynvml.nvmlInit()
    gpu_memory = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append((i, info.free)) # (index, free_bytes)
    
    # Sort by most free memory first
    sorted_gpus = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
    return [gpu[0] for gpu in sorted_gpus[:limit]]

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

#global vars for workers
worker_dino = None
worker_raft = None
worker_device = None
worker_transform = None

#init worker on specific gpu
def init_worker(gpu_queue):
    global worker_dino, worker_raft, worker_device, worker_transform
    gpu_id = gpu_queue.get()
    worker_device = torch.device(f"cuda:{gpu_id}")
    
    #load models once
    model_name = 'facebook/dinov2-small'
    worker_dino = AutoModel.from_pretrained(model_name, output_attentions=True).to(worker_device).eval()
    worker_raft = raft_large(pretrained=True, progress=False).to(worker_device).eval()
    
    #raft transform
    worker_transform = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
        T.Resize(size=(520, 960)),
    ])

#process a single pair of frames
def process_pair(args):
    i, img1_path, img2_path = args
    global worker_dino, worker_raft, worker_device, worker_transform
    
    #now, cutLER based on Dinov2 will give out 2 masks, for 2 salient objects
    masks = maskcut_method(img1_path, num_objects = 2, dino_model = worker_dino)
    
    mask1 = masks[0]
    mask2 = masks[1]
    h, w = mask1.shape
    
    #raft flow
    img1_t = read_image(img1_path)
    img2_t = read_image(img2_path)
    
    batch1 = worker_transform(img1_t).unsqueeze(0).to(worker_device)
    batch2 = worker_transform(img2_t).unsqueeze(0).to(worker_device)
    
    with torch.no_grad():
        flow_output = worker_raft(batch1, batch2)
        flow = flow_output[-1][0] 
        
    #camera motion cancellation (method 1: classical)
    #convert to numpy directly to save disk reads
    img1_np = img1_t.permute(1, 2, 0).numpy()[..., ::-1] 
    img2_np = img2_t.permute(1, 2, 0).numpy()[..., ::-1]
    
    camera_flow_np = get_camera_flow(img1_np, img2_np) #shape is (h, w, 2)
    camera_flow = torch.tensor(camera_flow_np).permute(2, 0, 1).to(worker_device)
    
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
    
    return i, torch.tensor(combined_score_mask1, dtype=torch.float32), torch.tensor(combined_score_mask2, dtype=torch.float32)

def process_video_fusion(video_folder, name, num_gpus=2):
    #get frame paths
    frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"no frames found in {video_folder}")
        
    print(f"processing {len(frame_paths)} frames for {name} on {num_gpus} gpus...")
    
    #setup gpu queue
    free_gpus = get_free_gpus(limit=num_gpus)
    m = mp.Manager()
    gpu_queue = m.Queue()
    for gid in free_gpus:
        gpu_queue.put(gid)
        
    #create arguments for parallel processing
    pairs = [(i, frame_paths[i], frame_paths[i+1]) for i in range(len(frame_paths) - 1)]
    
    results = []
    
    #parallel inference loop
    with ProcessPoolExecutor(max_workers=num_gpus, initializer=init_worker, initargs=(gpu_queue,)) as executor:
        for res in tqdm(executor.map(process_pair, pairs), total=len(pairs), desc="fusing pca and flow"):
            results.append(res)
            
    #sort results by original index
    results.sort(key=lambda x: x[0])
    
    fused_scores_mask1 = [r[1] for r in results]
    fused_scores_mask2 = [r[2] for r in results]
    
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
        #read original frame
        frame = cv2.imread(frame_paths[i])
        
        #convert heatmap to viridis-style colors
        heatmap = (fused_tensor[i].numpy() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
        
        #side-by-side
        combined = np.hstack((frame, heatmap_color))
        
        out.write(combined)
        
    out.release()
    print("video saved successfully.")


if __name__ == "__main__":
    #required for multiprocess spawn on clusters
    mp.set_start_method('spawn', force=True)

    # run config
    video_name = "swim"   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_folder = os.path.join(project_root, f"videos/{video_name}")
    
    #execute
    fused_tensor_mask1, fused_tensor_mask2 = process_video_fusion(target_folder, video_name)
    
    frame_paths = sorted(glob.glob(os.path.join(target_folder, "*.jpg")))
    save_fusion_video(frame_paths, fused_tensor_mask1, f"{video_name}_mask1")
    save_fusion_video(frame_paths, fused_tensor_mask2, f"{video_name}_mask2")