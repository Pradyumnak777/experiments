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
from depth_anything_3.api import DepthAnything3

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
    dino_transform = v2.Resize((224, 224), antialias=True)
    
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
    zero_flow = torch.zeros((1, 2, h, w), dtype=torch.half)
    flow_tensor_half = torch.cat(flow_list + [zero_flow], dim=0)

    #save files
    print(f"  # {vid_name}: saving tensors to disk...")
    torch.save(torch.stack(dino_list), os.path.join(video_dir, "dino.pt"))
    torch.save(flow_tensor_half, os.path.join(video_dir, "flow.pt")) 
    torch.save(depth_tensor_half, os.path.join(video_dir, "depth.pt"))
    
    print(f"done: {vid_name}")
    
    
class UCFRep_train(Dataset):
    def __init__(self, root_dir, clip_len=8, k_gap=5):
        '''
        root_dir: path to 'ucfrep_intermediate_dataset'
        clip_len: number of frames per clip (t)
        k_gap: temporal distance between anchor and positive clip
        '''
        self.root_dir = root_dir
        # filter out any non-directory files like .DS_Store
        self.video_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        self.clip_len = clip_len
        self.k_gap = k_gap
            
    def __len__(self):
        return len(self.video_folders)
    
    def _get_stacked_input(self, v_path, idx_range):
        # dino is now [t, 384, 16, 16]
        dino = torch.load(os.path.join(v_path, "dino.pt"), weights_only=True)[idx_range].float()
        flow = torch.load(os.path.join(v_path, "flow.pt"), weights_only=True)[idx_range].float()
        depth = torch.load(os.path.join(v_path, "depth.pt"), weights_only=True)[idx_range].float()
        
        # interpolate from 16x16 up to 224x224
        dino = F.interpolate(dino, size=(224, 224), mode='bilinear', align_corners=False) 
        
        return torch.cat([dino, flow, depth], dim=1)
    
    def __getitem__(self, idx):
        vid_name = self.video_folders[idx]
        v_path = os.path.join(self.root_dir, vid_name)
        
        # use dino.pt to check the total frame count of the video
        dino_meta = torch.load(os.path.join(v_path, "dino.pt"), weights_only=True)
        t_total = dino_meta.shape[0]

        # ensure we have enough frames for the anchor + gap + positive clip
        max_start = t_total - self.clip_len - self.k_gap - 1
        
        if max_start <= 0:
            # fallback for very short videos
            start_t = 0
            k = 0
        else:
            # random starting point for the anchor clip
            start_t = torch.randint(0, max_start, (1,)).item()
            k = self.k_gap

        # anchor and positive slices
        anchor_slice = slice(start_t, start_t + self.clip_len)
        positive_slice = slice(start_t + k, start_t + k + self.clip_len)

        anchor_clip = self._get_stacked_input(v_path, anchor_slice)     #[t, 387, 224, 224]
        positive_clip = self._get_stacked_input(v_path, positive_slice) #[t, 387, 224, 224]

        return {
            "anchor": anchor_clip,
            "positive": positive_clip,
            "vid_name": vid_name
        }
    
if __name__ == "__main__":
    dir = "UCF_Rep/train"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #depth model - switch to base for speed
    depth_model = DepthAnything3.from_pretrained("depth-anything/da3-base")
    depth_model = depth_model.to(device=device)
    
    #dino model
    model_name = 'facebook/dinov2-small'
    dino_model = AutoModel.from_pretrained(model_name, output_attentions=True).cuda()
    dino_model.eval()
    
    #optical flow model
    raft_model = raft_large(pretrained=True, progress=False).cuda().eval()

    for idx, vid_file in enumerate(os.listdir(dir)):
        #each one is an mp4 file
        frames = mp4_to_frames(os.path.join(dir, vid_file)) #get frames
        video_tensor = tensorize_vid(frames, transform)
        
        preprocess(video_tensor, frames, vid_file, dino_model, raft_model, depth_model)