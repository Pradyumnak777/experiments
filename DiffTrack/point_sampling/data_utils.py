import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision.transforms import v2
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dino_exp import maskcut_tensor_method #for dino
from utils.depth_exp import get_frame_depth #for depth
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
        # CONVERT BGR TO RGB
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
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # <-- Fixed here
    v2.Resize(size=(520, 960)),
])

def tensorize_vid(frames, transform):
    #to return a 4d tensor - [T, C, H, W]
    video_tensor = torch.stack([v2.functional.to_image(f) for f in frames])
    
    #the new v2 transform seems to be able to apply transforms to videos/batches..
    return transform(video_tensor)

def preprocess(vid_tensor, raw_frames, name, dino_model, raft_model, depth_model = None):
    '''
    input format: [f0_tensor, f1_tensor, .., fn_tensor] -> (T, C, H, W)
    output format: save on disk like-
    
    dataset/
        - video_001/
            - dino.pt
            - flow.pt
            - depth.pt
            - mask0.pt
            - mask1.pt
        - video_002/
            - ..
    '''
    t, c, h, w = vid_tensor.shape
    vid_name = os.path.splitext(name)[0]

    base_dir = "ucfrep_intermediate_dataset"
    video_dir = os.path.join(base_dir, vid_name)
    os.makedirs(video_dir, exist_ok=True)

    # target file paths for this video
    dino_path = os.path.join(video_dir, "dino.pt")
    flow_path = os.path.join(video_dir, "flow.pt")
    depth_path = os.path.join(video_dir, "depth.pt")
    mask0_path = os.path.join(video_dir, "mask0.pt")
    mask1_path = os.path.join(video_dir, "mask1.pt")
    
    # storage for the full video stack
    dino_list, flow_list, depth_list = [], [], []
    m0_list, m1_list = [], []
    
    for i in range(t):
        #get current frame and add batch dim
        frame_t = vid_tensor[i].unsqueeze(0).cuda()
        #get next one also for raft
        # frame_t_next = vid_tensor[i+1].unsqueeze(0).cuda()
        #1. masks and dino from maskcut
        masks, dino_feats = maskcut_tensor_method(img_tensor=frame_t, num_objects=2, dino_model=dino_model) #get 2 masks
        # mask0 = torch.tensor(masks[0]).cuda() # [h, w]
        # mask1 = torch.tensor(masks[1]).cuda() # [h, w]
        # dino = torch.tensor(dino_feats).cuda() #[384, h, w]
        m0_list.append(torch.tensor(masks[0]).cpu())
        m1_list.append(torch.tensor(masks[1]).cpu())
        dino_list.append(dino_feats.cpu()) #[384, H, W]
        
        #now do raft
        if i < t - 1:
            curr_img = v2.functional.to_image(raw_frames[i])
            next_img = v2.functional.to_image(raw_frames[i+1])
            
            raft_f_curr = raft_transform(curr_img).unsqueeze(0).cuda()
            raft_f_next = raft_transform(next_img).unsqueeze(0).cuda()
            
            with torch.no_grad():
                flow_output = raft_model(raft_f_curr, raft_f_next)
                flow = flow_output[-1][0]  #[2, h , w]
                flow = F.interpolate(flow.unsqueeze(0), size=(h, w), mode="bilinear").squeeze(0)
            flow_list.append(flow.cpu())
        
        #now do depth
        depth = get_frame_depth(raw_frames[i], depth_model)
        depth_list.append(depth.cpu())
    
    torch.save(torch.stack(dino_list), os.path.join(video_dir, "dino.pt"))
    torch.save(torch.stack(flow_list), os.path.join(video_dir, "flow.pt"))
    torch.save(torch.stack(depth_list), os.path.join(video_dir, "depth.pt"))
    torch.save(torch.stack(m0_list), os.path.join(video_dir, "mask0.pt"))
    torch.save(torch.stack(m1_list), os.path.join(video_dir, "mask1.pt"))
    
    print(f"preproc'd and saved: {vid_name}")


class UCFRep_train(Dataset):
    def __init__(self, root_dir, clip_len=8, k_gap=5):
        '''
        root_dir: path to 'ucfrep_intermediate_dataset'
        clip_len: number of frames per clip (T)
        k_gap: temporal distance between anchor and positive clip (ts is how much to "look ahead")
        '''
        self.root_dir = root_dir
        self.video_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        self.clip_len = clip_len
        self.k_gap = k_gap
            
    def __len__(self):
        return len(self.video_folders)
    
    def _get_stacked_input(self, v_path, idx_range):
        '''
        for formatting into the input embedding
        '''
        dino = torch.load(os.path.join(v_path, "dino.pt"), weights_only=True)[idx_range]   # [T, 384, H, W]
        flow = torch.load(os.path.join(v_path, "flow.pt"), weights_only=True)[idx_range]   # [T, 2, H, W]
        depth = torch.load(os.path.join(v_path, "depth.pt"), weights_only=True)[idx_range] # [T, 1, H, W]
        m0 = torch.load(os.path.join(v_path, "mask0.pt"), weights_only=True)[idx_range]     # [T, H, W]
        m1 = torch.load(os.path.join(v_path, "mask1.pt"), weights_only=True)[idx_range]     # [T, H, W]
        
        #making sure all same dims -> [T, 1, H, W]
        m0 = m0.unsqueeze(1)
        m1 = m1.unsqueeze(1)
        
        stacked = torch.cat([dino, flow, depth, m0, m1], dim=1)
        return stacked
    
    def __getitem__(self, idx):
        vid_name = self.video_folders[idx]
        v_path = os.path.join(self.root_dir, vid_name)
        
        #justo check num_frames in this video/clip
        temp_meta = torch.load(os.path.join(v_path, "mask0.pt"), weights_only=True)
        t_total = temp_meta.shape[0]

        max_start = t_total - self.clip_len - self.k_gap - 1
        
        if max_start <= 0:
            start_t = 0
            k = 0
        else:
            #for anchor clip
            start_t = torch.randint(0, max_start, (1,)).item()
            k = self.k_gap

        #get positive and anchor from same clip. negative will be taken from the batch
        anchor_slice = slice(start_t, start_t + self.clip_len)
        positive_slice = slice(start_t + k, start_t + k + self.clip_len) #at k time away

        anchor_clip = self._get_stacked_input(v_path, anchor_slice)     # [T, 389, H, W]
        positive_clip = self._get_stacked_input(v_path, positive_slice) # [T, 389, H, W]

        return {
            "anchor": anchor_clip,
            "positive": positive_clip,
            "vid_name": vid_name
        }

if __name__ == "__main__":
    dir = "UCF_Rep/train"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #depth model
    depth_model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
    depth_model = depth_model.to(device=device)
    
    #dino model
    model_name = 'facebook/dinov2-small'
    dino_model = AutoModel.from_pretrained(model_name, output_attentions=True).cuda()
    dino_model.eval()
    
    #optical flow model
    raft_model = raft_large(pretrained=True, progress=False).cuda().eval()

    # train_dict = {}
    for idx, vid_file in enumerate(os.listdir(dir)):
        #each one is an mp4 file
        frames = mp4_to_frames(os.path.join(dir, vid_file)) #get frames
        # train_dict[f"video_{idx}"] = frames
        video_tensor = tensorize_vid(frames, transform)
        
        preprocess(video_tensor, frames, vid_file, dino_model, raft_model, depth_model)
        
    #we now have the intermediates files stored on disk

    

        