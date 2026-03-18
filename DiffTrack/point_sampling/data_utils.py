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
    zero_flow = torch.zeros((1, 2, h, w), dtype=torch.half)
    flow_tensor_half = torch.cat(flow_list + [zero_flow], dim=0)

    #save files
    print(f"  # {vid_name}: saving tensors to disk...")
    torch.save(torch.stack(dino_list), os.path.join(video_dir, "dino.pt"))
    torch.save(flow_tensor_half, os.path.join(video_dir, "flow.pt")) 
    torch.save(depth_tensor_half, os.path.join(video_dir, "depth.pt"))
    
    print(f"done: {vid_name}")
    
    
# class UCFRep_train(Dataset):
#     def __init__(self, root_dir, clip_len=8, k_gap=5):
#         '''
#         root_dir: path to 'ucfrep_intermediate_dataset'
#         clip_len: number of frames per clip (t)
#         k_gap: temporal distance between anchor and positive clip
#         '''
#         self.root_dir = root_dir
#         # filter out any non-directory files like .DS_Store
#         self.video_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
#         self.clip_len = clip_len
#         self.k_gap = k_gap
            
#     def __len__(self):
#         return len(self.video_folders)
    
#     def _get_stacked_input(self, v_path, idx_range):
#         # dino is now [t, 384, 16, 16]
#         dino = torch.load(os.path.join(v_path, "dino.pt"), weights_only=True)[idx_range].float()
#         flow = torch.load(os.path.join(v_path, "flow.pt"), weights_only=True)[idx_range].float()
#         depth = torch.load(os.path.join(v_path, "depth.pt"), weights_only=True)[idx_range].float()
        
#         # interpolate from 16x16 up to 224x224
#         dino = F.interpolate(dino, size=(224, 224), mode='bilinear', align_corners=False) 
        
#         return torch.cat([dino, flow, depth], dim=1)
    
#     def __getitem__(self, idx):
#         vid_name = self.video_folders[idx]
#         v_path = os.path.join(self.root_dir, vid_name)
        
#         # use dino.pt to check the total frame count of the video
#         dino_meta = torch.load(os.path.join(v_path, "dino.pt"), weights_only=True)
#         t_total = dino_meta.shape[0]

#         # ensure we have enough frames for the anchor + gap + positive clip
#         max_start = t_total - self.clip_len - self.k_gap - 1
        
#         if max_start <= 0:
#             # fallback for very short videos
#             start_t = 0
#             k = 0
#         else:
#             # random starting point for the anchor clip
#             start_t = torch.randint(0, max_start, (1,)).item()
#             k = self.k_gap

#         # anchor and positive slices [FROM SAME VIDEO!!!]
#         anchor_slice = slice(start_t, start_t + self.clip_len)
#         positive_slice = slice(start_t + k, start_t + k + self.clip_len)

#         anchor_clip = self._get_stacked_input(v_path, anchor_slice)     #[t, 387, 224, 224]
#         positive_clip = self._get_stacked_input(v_path, positive_slice) #[t, 387, 224, 224]
        
#         #for negative 2 things can be done-
#         '''
#         1. within the frame/image, treat all pixels outside the mask as negative (this is from the generated output)
#         2. inside the batch, treat videos with a different vid_name as negatives. And then treat the predicted mask of this
#         clip from another video as negative.
#         '''

#         return {
#             "anchor": anchor_clip,
#             "positive": positive_clip,
#             "vid_name": vid_name
#         }
        
# import torch
# from torch.utils.data import Dataset
# import os
# import cv2
# import torch.nn.functional as F

# class UCFRep_Finetune_Dataset(Dataset):
#     def __init__(self, mp4_dir, pt_dir, clip_len=8, k_gap=5):
#         '''
#         mp4_dir: path to 'UCF_Rep/train' (the raw videos)
#         pt_dir: path to 'ucfrep_intermediate_dataset' (where flow/depth are)
#         '''
#         self.mp4_dir = mp4_dir
#         self.pt_dir = pt_dir
        
#         self.video_names = [f for f in os.listdir(pt_dir) if os.path.isdir(os.path.join(pt_dir, f))]
#         self.clip_len = clip_len
#         self.k_gap = k_gap
        
#         self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#     def __len__(self):
#         return len(self.video_names)

#     def _extract_pixel_clip(self, video_path, idx_range):
#         frames = []
#         cap = cv2.VideoCapture(video_path)
        
#         #jump to the start frame
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx_range.start)
        
#         total_frames = idx_range.stop - idx_range.start
#         expected_len = total_frames // idx_range.step
        
#         for i in range(total_frames):
#             ret, frame = cap.read()
#             #if opencv hits the end of the file early, stop reading
#             if not ret: 
#                 break
                
#             #only grab frames that match our stride
#             if i % idx_range.step == 0:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, (224, 224))
#                 t_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
#                 t_frame = (t_frame - self.norm_mean) / self.norm_std
#                 frames.append(t_frame)
        
#         cap.release()
        
#         #if it read absolutely nothing (bad seek), fill with zeros
#         if len(frames) == 0:
#             return torch.zeros((expected_len, 3, 224, 224))
            
#         #if it read some frames but hit EOF before getting all 8, duplicate the last frame
#         while len(frames) < expected_len:
#             frames.append(frames[-1].clone())
            
#         return torch.stack(frames) #now guaranteed to return [8, 3, 224, 224]

#     def __getitem__(self, idx):
#         #fetch the related files for this video file (video, flow.pt, depth.pt)
#         v_name = self.video_names[idx]
#         pt_path = os.path.join(self.pt_dir, v_name)
#         mp4_path = os.path.join(self.mp4_dir, v_name + ".mp4")

#         #get total frames
#         flow_meta = torch.load(os.path.join(pt_path, "flow.pt"), weights_only=True)
#         t_total = flow_meta.shape[0]

#         # Calculate temporal slices
#         max_start = t_total - self.clip_len - self.k_gap - 1 
#         if max_start <= 0: #edge case, shouldnt happen..
#             start_t, k = 0, 0
#         else:
#             start_t = torch.randint(0, max_start, (1,)).item() #chose csome random stanrting index
#             k = self.k_gap # k is how far ahead to look in number of frames to find the poitive pair

#         anchor_idx = slice(start_t, start_t + self.clip_len)
#         pos_idx = slice(start_t + k, start_t + k + self.clip_len) #positive clip is curr_frame + k

        
#         #multiply start_t by 2 because .pt files were created with stride=2
#         anchor_pixels = self._extract_pixel_clip(mp4_path, slice(start_t*2, (start_t + self.clip_len)*2, 2))
#         pos_pixels = self._extract_pixel_clip(mp4_path, slice((start_t+k)*2, (start_t+k+self.clip_len)*2, 2))

#         #get flow and depth
#         anchor_flow = torch.load(os.path.join(pt_path, "flow.pt"), weights_only=True)[anchor_idx].float()
#         anchor_depth = torch.load(os.path.join(pt_path, "depth.pt"), weights_only=True)[anchor_idx].float()
        
#         pos_flow = torch.load(os.path.join(pt_path, "flow.pt"), weights_only=True)[pos_idx].float()
#         pos_depth = torch.load(os.path.join(pt_path, "depth.pt"), weights_only=True)[pos_idx].float()

#         return {
#             "anchor_pixels": anchor_pixels, # [T, 3, 224, 224] -> Into DINO
#             "pos_pixels": pos_pixels,       # [T, 3, 224, 224] -> Into DINO
#             "anchor_flow": anchor_flow,     # [T, 2, 224, 224] -> Mask Guidance
#             "anchor_depth": anchor_depth,   # [T, 1, 224, 224] -> Scale Guidance
#             "pos_flow": pos_flow,
#             "pos_depth": pos_depth,
#             "vid_name": v_name
#         }
    
    
class UCFRep_finetune(Dataset):
    def __init__(self, k = 6, mp4_dir = None, pt_dir = None): #strided dataset!! frame rate is halved!!
        '''
        mp4_dir: path to 'UCF_Rep/train' (the raw videos)
        pt_dir: path to 'ucfrep_intermediate_dataset' (where flow/depth are- at halved framerate)
        '''
        self.mp4_dir = mp4_dir
        self.video_names = [f for f in os.listdir(pt_dir) if os.path.isdir(os.path.join(pt_dir, f))]
        self.k = k
        
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __len__(self): #should i return num of videos? or the num of total imag epairs i'll form from all the videos..?
        return len(self.video_names)
        #note: __len__ defines how many times the __getitem__ method can be called per epoch..
    
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
        '''
        to return:
        1. video(RGB), flow, depth (entire video)
        
        OR
        
        2. [2, 3, H, W] , [2, 2, H, W] , [2, 1, H, W] -> returning only the pair
        '''
        v_name = self.video_names[idx]
        pt_path = os.path.join(self.pt_dir, v_name)
        mp4_path = os.path.join(self.mp4_dir, v_name + ".mp4")
        
        #note that these are strided by 2..
        flow_all = torch.load(os.path.join(pt_path, "flow.pt"), weights_only=True)
        depth_all = torch.load(os.path.join(pt_path, "depth.pt"), weights_only=True)
        t_subsampled = flow_all.shape[0] #num of frames
        
        #now a pair, (t, t+k) is needed. again note that k =6, mens actually 12, as frames are subsampled
        max_start = t_subsampled - self.k - 1
        if max_start <= 0: #if clip is very short
            im1_idx = 0
            im2_idx = min(self.k, t_subsampled - 1)
        else: #pick random
            im1_idx = torch.randint(0, max_start, (1,)).item()
            im2_idx = im1_idx + self.k
            
        im1_flow = flow_all[im1_idx].float()   #[2, 224, 224]
        im1_depth = depth_all[im1_idx].float()   #[1, 224, 224]
        
        im2_flow = flow_all[im2_idx].float()   #[2, 224, 224]
        im2_depth = depth_all[im2_idx].float()   #[1, 224, 224]
        
        im1 = self._get_single_frame(mp4_path, im1_idx * 2) #multiplying by 2 as stride was 2. So keeping every 2nd frame
        im2 = self._get_single_frame(mp4_path, im2_idx * 2)
        
        return {
            "im1_pixels": im1,
            "im2_pixels": im2, 
            "im1_flow": im1_flow,
            "im1_depth": im1_depth,
            "im2_flow": im2_flow,
            "im2_depth": im2_depth,
            "vid_name": v_name
        }
    
if __name__ == "__main__":
    dir = "UCF_Rep/train"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #depth model - switch to base for speed
    depth_model = DepthAnything3.from_pretrained("depth-anything/da3-base")
    depth_model = depth_model.to(device=device)
    
    #dino model
    model_name = 'facebook/dinov2-base'
    dino_model = AutoModel.from_pretrained(model_name, output_attentions=True).cuda()
    dino_model.eval()
    
    #optical flow model
    raft_model = raft_large(pretrained=True, progress=False).cuda().eval()

    for idx, vid_file in enumerate(os.listdir(dir)):
        #each one is an mp4 file
        frames = mp4_to_frames(os.path.join(dir, vid_file)) #get frames
        video_tensor = tensorize_vid(frames, transform)
        
        preprocess(video_tensor, frames, vid_file, dino_model, raft_model, depth_model)