import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision.transforms import v2
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dino_exp import get_dino_features #for dino
from utils.depth_exp import get_batch_depth #for depth
from torchvision.models.optical_flow import raft_large #for optical flow
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np

def mp4_to_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #convert bgr to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames #list of img frames

transform = v2.Compose([
    v2.ToImage(), #tensorizes
    v2.ToDtype(torch.float32, scale=True), #scales to [0, 1]
    v2.Resize((224, 224), antialias=True), #parallel resize
    v2.CenterCrop(224),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406], #imagenet stats
        std=[0.229, 0.224, 0.225]
    )    
])

raft_transform = v2.Compose([
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    v2.Resize(size=(520, 960)),
])

def tensorize_vid(frames, transform):
    #to return a 4d tensor - [t, c, h, w]
    video_tensor = torch.stack([v2.functional.to_image(f) for f in frames])
    return transform(video_tensor)

def preprocess(vid_tensor, raw_frames, name, dino_model, raft_model, depth_model=None, stride=2):
    t, c, h, w = vid_tensor.shape
    vid_name = os.path.splitext(name)[0]
    video_dir = os.path.join("ucfrep_intermediate_dataset", vid_name)
    
    if os.path.exists(os.path.join(video_dir, "depth.pt")):
        print(f"skipping: {vid_name} (already processed)")
        return
    
    os.makedirs(video_dir, exist_ok=True)

    dino_list = []
    
    #stride applied to raw frames for depth and flow
    #this creates the 'sparse' video we actually store
    strided_raw_frames = raw_frames[::stride]
    strided_vid_tensor = vid_tensor[::stride]
    t_sparse = len(strided_raw_frames)

    print(f"  # {vid_name}: extracting raw dino features (stride={stride})...")
    dino_transform = v2.Compose([
        v2.Resize((518, 518), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i in range(t_sparse):
        #normalize 
        frame_t = dino_transform(strided_vid_tensor[i].unsqueeze(0))
        
        #just get the features
        dino_feats = get_dino_features(frame_t, dino_model)
        dino_list.append(dino_feats)
        
    #2. depth with chunking
    print(f"  # {vid_name}: running batched depth...")
    depth_tensor_half = get_batch_depth(strided_raw_frames, depth_model, target_size=(h, w), chunk_size=16)
    
    #3. raft flow with chunking
    print(f"  # {vid_name}: running batched raft flow...")
    flow_list = []
    raft_chunk = 8 
    raft_device = next(raft_model.parameters()).device
    
    #calculate flow between the sparse frames
    for i in range(0, t_sparse - 1, raft_chunk):
        end_idx = min(i + raft_chunk, t_sparse - 1)
        curr_chunk = [strided_raw_frames[j] for j in range(i, end_idx)]
        next_chunk = [strided_raw_frames[j+1] for j in range(i, end_idx)]
        
        curr_t = torch.stack([raft_transform(v2.functional.to_image(f)) for f in curr_chunk]).to(raft_device)
        next_t = torch.stack([raft_transform(v2.functional.to_image(f)) for f in next_chunk]).to(raft_device)
        
        with torch.no_grad():
            outputs = raft_model(curr_t, next_t)
            flow = F.interpolate(outputs[-1], size=(h, w), mode="bilinear")
            flow_list.append(flow.half().cpu())
            
        torch.cuda.empty_cache()

    #combine and pad last frame
    zero_flow = torch.zeros((1, 2, h, w), dtype=torch.half) #last entry is zero flow
    flow_tensor_half = torch.cat(flow_list + [zero_flow], dim=0)

    #save files
    print(f"  # {vid_name}: saving tensors to disk...")
    torch.save(torch.stack(dino_list), os.path.join(video_dir, "dino.pt"))
    torch.save(flow_tensor_half, os.path.join(video_dir, "flow.pt")) 
    torch.save(depth_tensor_half, os.path.join(video_dir, "depth.pt"))
    
    print(f"done: {vid_name}")
    
class UCFRep_finetune(Dataset):
    def __init__(self, mp4_dir = None, pt_dir = None): 
        '''
        mp4_dir: path to 'UCF_Rep/train' (the raw videos)
        pt_dir: path to 'ucfrep_intermediate_dataset' (where flow/depth are- at halved framerate)
        '''
        self.mp4_dir = mp4_dir
        self.pt_dir = pt_dir
        
        # we loop over the videos directly now
        self.video_names = []
        if os.path.exists(pt_dir):
            for v_folder in os.listdir(pt_dir):
                if os.path.isdir(os.path.join(pt_dir, v_folder)):
                    self.video_names.append(v_folder)
        
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __len__(self): 
        return len(self.video_names)
    
    def _get_single_frame(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return torch.zeros((3, 224, 224))
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        t_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return (t_frame - self.norm_mean) / self.norm_std
    
    def __getitem__(self, idx):
            vid_name = self.video_names[idx]
            pt_path = os.path.join(self.pt_dir, vid_name)
            mp4_path = os.path.join(self.mp4_dir, vid_name + ".mp4")
            
            #load just the shapes to find how many frames we have
            t_frames = torch.load(os.path.join(pt_path, "dino.pt"), map_location='cpu', weights_only=True).shape[0]
            
            chunk_size = 3 #grabbing 3 consecutive strided frames
            
            #pick a random start frame, leaving room for the chunk
            start_idx = torch.randint(0, max(1, t_frames - chunk_size), (1,)).item()
            
            #load the full tensors (it's faster to load all and slice than load individual items)
            all_flow = torch.load(os.path.join(pt_path, "flow.pt"), weights_only=True).float()
            all_depth = torch.load(os.path.join(pt_path, "depth.pt"), weights_only=True).float()
            
            chunk_pixels = []
            chunk_flow = []
            chunk_depth = []
            
            for i in range(start_idx, start_idx + chunk_size):
                #multiply by 2 because stride was 2
                frame = self._get_single_frame(mp4_path, i * 2)
                chunk_pixels.append(frame)
                chunk_flow.append(all_flow[i])
                chunk_depth.append(all_depth[i])
                
            return {
                "pixels": torch.stack(chunk_pixels), #[3, 3, 224, 224]
                "flow": torch.stack(chunk_flow),     #[3, 2, 224, 224]
                "depth": torch.stack(chunk_depth),   #[3, 1, 224, 224]
                "vid_name": vid_name
            }            
