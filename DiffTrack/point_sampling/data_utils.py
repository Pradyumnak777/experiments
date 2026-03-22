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
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

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
    
# renamed to intra_video and updated logic to pick pairs from the same video    
def save_graph_intra_video(vid_name, frames, pt_dir, device, dino_model, cls_name, stride=2, num_pairs=3):
    strided = frames[::stride] #keeping only every second frame
    n_frames = len(strided)
    
    if n_frames < 2:
        return # video too short to make pairs
        
    # load the precomputed dino features to instantly find visually distinct pairs
    # shape: [t, 768, 16, 16]
    dino_pt_path = os.path.join(pt_dir, vid_name, "dino.pt")
    if not os.path.exists(dino_pt_path):
        return
        
    saved_feats = torch.load(dino_pt_path, map_location='cpu', weights_only=True)
    
    # average spatially to get a 1d summary vector per frame [t, 768]
    global_feats = saved_feats.mean(dim=(2, 3)) 
    
    # calculate full similarity matrix for this video [t, t]
    sim_matrix = F.cosine_similarity(global_feats.unsqueeze(1), global_feats.unsqueeze(0), dim=-1)
    
    # pick 'num_pairs' random starting frames
    idx_list = torch.randperm(n_frames)[:num_pairs]
    
    for p in range(len(idx_list)):
        i = idx_list[p].item()
        
        # look at how similar frame 'i' is to all other frames
        sims = sim_matrix[i]
        
        # get the average similarity. we want frames below this, but not completely different (<0.4)
        avg_sim = sims.mean().item()
        
        #"visually distinct" yet sematically similar..
        lower_bound = avg_sim * 0.8
        upper_bound = avg_sim * 1.3
        valid_j_indices = torch.where((sims >= lower_bound) & (sims <= upper_bound))[0]
        
        if len(valid_j_indices) == 0:
            # fallback: just pick a frame somewhat far away temporally
            j = (i + n_frames // 3) % n_frames
        else:
            # randomly pick one of our visually distinct frames
            j = valid_j_indices[torch.randint(0, len(valid_j_indices), (1,))].item()
        
        # skip if we accidentally matched identical frames
        if i == j: continue 
        
        im1, im2 = strided[i], strided[j]
        
        #now perform the joint co-segmentation on these images(like in the paper - "wholly unsupervised!")
        res_size = 518 #DINOv2-base resolution (37x37 grid)
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((res_size, res_size), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        img1_t = transform(im1).unsqueeze(0).to(device)
        img2_t = transform(im2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            #extract dino feats
            feat1 = dino_model(img1_t).last_hidden_state[0, 1:, :] #[1369, 768]
            feat2 = dino_model(img2_t).last_hidden_state[0, 1:, :] # 1369, 768]
            
            feat1 = F.normalize(feat1, p=2, dim=-1)
            feat2 = F.normalize(feat2, p=2, dim=-1)
        
        #join these features for the graph- [2378, 768]
        joint_feats = torch.cat([feat1, feat2], dim=0)
        S = torch.matmul(joint_feats, joint_feats.T) #this is basically a graph of the cosine similarities b/w each feature(normalize dbefore too)
        
        '''
        using the method used in the 'wholly unsupervised!' paper..
        '''
        sigma_a = 0.4
        sigma_r = 0.3
        omega = 0.2
        
        A = torch.exp(-((1.0 - S)**2) / (2 * sigma_a**2)) #attraction
        R = torch.exp(-((S + 1.0)**2) / (2 * sigma_r**2)) #repulsion
        A = A.cpu().numpy()
        R = R.cpu().numpy()
        
        R = omega * R
        
        #the graph
        D_A = np.diag(np.sum(A, axis=1))
        D_R = np.diag(np.sum(R, axis=1))
        
        W = A - R + D_R
        D = D_A + D_R
        L = D - W
        
        try:
            eigval, eigvec = linalg.eigsh(L, k=2, which='SM', M=D)
            z = eigvec[:, 1]
            
            split_val = np.mean(z)
            mask_raw = (z > split_val).astype(np.float32)
            
            #spllitting back to the frames
            num_patches = feat1.shape[0]
            mask1_raw_check = mask_raw[:num_patches].reshape(37, 37)
            
            #if the majority of corners are '1', 
            #it means the mask accidentally labeled the background as the actor.
            corner_sum = mask1_raw_check[0,0] + mask1_raw_check[0,-1] + mask1_raw_check[-1,0] + mask1_raw_check[-1,-1]
            
            if corner_sum > 2:
                #invert the entire joint mask so the actor becomes 1
                joint_mask_final = 1.0 - mask_raw
            else:
                joint_mask_final = mask_raw
                
            mask1_raw = joint_mask_final[:num_patches].reshape(37, 37)
            mask2_raw = joint_mask_final[num_patches:].reshape(37, 37)
                
        except Exception as e:
            mask1_raw = np.zeros((37, 37), dtype=np.float32)
            mask2_raw = np.zeros((37, 37), dtype=np.float32)
    
        #save the masks inside a class folder so it's easy to track
        base_dir = os.path.join("co-segmentation", cls_name)
        # updated pair name format to reflect it's the same video
        pair_name = f"{vid_name}_{i:04d}_AND_{j:04d}"
        pair_dir = os.path.join(base_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)

        np.save(os.path.join(pair_dir, "mask1.npy"), mask1_raw.astype(np.float32))
        np.save(os.path.join(pair_dir, "mask2.npy"), mask2_raw.astype(np.float32))
        
        #VISUALIZATION BLOCK
        #creating a 2x2 grid: frames vs. masks
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        #frame 1
        axes[0, 0].imshow(im1)
        axes[0, 0].set_title(f"anchor ({vid_name} f{i*stride})")
        axes[0, 0].axis('off')
        
        #mask 1
        axes[0, 1].imshow(im1)
        axes[0, 1].imshow(cv2.resize(mask1_raw, (im1.shape[1], im1.shape[0])), alpha=0.5, cmap='jet')
        axes[0, 1].set_title("joint mask 1")
        axes[0, 1].axis('off')
        
        #frame 2
        axes[1, 0].imshow(im2)
        axes[1, 0].set_title(f"positive ({vid_name} f{j*stride})")
        axes[1, 0].axis('off')
        
        #mask 2
        axes[1, 1].imshow(im2)
        axes[1, 1].imshow(cv2.resize(mask2_raw, (im2.shape[1], im2.shape[0])), alpha=0.5, cmap='jet')
        axes[1, 1].set_title("joint mask 2")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(pair_dir, "visual_check.jpg"), dpi=150)
        plt.close(fig)

    
class UCFRep_finetune(Dataset):
    def __init__(self, mp4_dir = None, pt_dir = None, co_seg_dir = "co-segmentation"): 
        '''
        mp4_dir: path to 'UCF_Rep/train' (the raw videos)
        pt_dir: path to 'ucfrep_intermediate_dataset' (where flow/depth are- at halved framerate)
        '''
        self.mp4_dir = mp4_dir
        self.pt_dir = pt_dir
        self.co_seg_dir = co_seg_dir
        
        #don't loop by video anymore, we loop by our pre-computed pairs
        self.pair_paths = []
        if os.path.exists(co_seg_dir):
            for cls_name in os.listdir(co_seg_dir):
                cls_path = os.path.join(co_seg_dir, cls_name)
                if os.path.isdir(cls_path):
                    for pair_folder in os.listdir(cls_path):
                        self.pair_paths.append(os.path.join(cls_path, pair_folder))
        
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __len__(self): 
        #returns total number of intra-video pairs we generated
        return len(self.pair_paths)
    
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
        returning the exact intra-video pair and their respective masks
        '''
        pair_path = self.pair_paths[idx]
        folder_name = os.path.basename(pair_path)
        
        #format is now vid_name_000idx_AND_000jdx
        parts = folder_name.split("_AND_")
        
        #extracting names and indices cleanly
        vid_parts = parts[0].split("_")
        vid_name = "_".join(vid_parts[:-1]) # rebuilds the video name cleanly
        im1_idx = int(vid_parts[-1])
        im2_idx = int(parts[1]) # frame 2 is just the second half of the split
        
        #setting up paths (only need one set now since it's the same video)
        pt_path = os.path.join(self.pt_dir, vid_name)
        mp4_path = os.path.join(self.mp4_dir, vid_name + ".mp4")
        
        #load flow and depth from pt dirs
        im1_flow = torch.load(os.path.join(pt_path, "flow.pt"), weights_only=True)[im1_idx].float()
        im1_depth = torch.load(os.path.join(pt_path, "depth.pt"), weights_only=True)[im1_idx].float()
        
        im2_flow = torch.load(os.path.join(pt_path, "flow.pt"), weights_only=True)[im2_idx].float()
        im2_depth = torch.load(os.path.join(pt_path, "depth.pt"), weights_only=True)[im2_idx].float()
        
        #load frames (multiplying by 2 as stride was 2)
        im1 = self._get_single_frame(mp4_path, im1_idx * 2) 
        im2 = self._get_single_frame(mp4_path, im2_idx * 2)
        
        #load the ccg masks we generated offline
        mask1 = torch.from_numpy(np.load(os.path.join(pair_path, "mask1.npy")))
        mask2 = torch.from_numpy(np.load(os.path.join(pair_path, "mask2.npy")))
        
        return {
            "im1_pixels": im1,
            "im2_pixels": im2, 
            "im1_flow": im1_flow,
            "im1_depth": im1_depth,
            "im2_flow": im2_flow,
            "im2_depth": im2_depth,
            "mask1": mask1,
            "mask2": mask2,
            "vid1_name": vid_name,
            "vid2_name": vid_name # kept this for downstream compatibility
        }        
            
if __name__ == "__main__":
    dir = "UCF_Rep/train"
    pt_dir = "ucfrep_intermediate_dataset"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #dino model
    model_name = 'facebook/dinov2-base'
    dino_model = AutoModel.from_pretrained(model_name, output_attentions=True).cuda()
    dino_model.eval()
    
    vid_files = [f for f in os.listdir(dir) if f.endswith('.mp4')]
    
    # loop through all videos independently
    for v_file in vid_files:
        vid_name = os.path.splitext(v_file)[0]
        cls_name = v_file.split('_')[1] 
        
        frames = mp4_to_frames(os.path.join(dir, v_file))
        
        # generate 3 random pairs entirely within the same video
        save_graph_intra_video(vid_name, frames, pt_dir, device, dino_model, cls_name, num_pairs=3)
        print(f"co-segmentation done for {vid_name}")