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
    
    
def save_graph_cross_video(vid1_name, vid2_name, frames1, frames2, device, dino_model, cls_name, stride=2, num_pairs=3):
    strided1 = frames1[::stride] #keeping only every second frame
    strided2 = frames2[::stride]
    
    idx_list1 = torch.randperm(len(strided1))[:num_pairs]
    idx_list2 = torch.randperm(len(strided2))[:num_pairs]
    
    #generate a few pairs per video combo to save processing time later
    for p in range(num_pairs):
        #pull from our unique lists instead of rolling the dice every time
        i = idx_list1[p].item()
        j = idx_list2[p].item()
        
        im1, im2 = strided1[i], strided2[j]
        
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
        sigma_a = 0.16
        sigma_r = 0.3
        omega = 0.4
        
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
            
            # #tokencut below
            # split_val = np.mean(z)
            # set_A = z <= split_val
            # set_B = z > split_val
            # max_idx = np.argmax(np.abs(z))
            
            # if set_A[max_idx]:
            #     joint_mask_raw = set_A.astype(np.float32)
            # else:
            #     joint_mask_raw = set_B.astype(np.float32)
            
            
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
        pair_name = f"{vid1_name}_{i:04d}_AND_{vid2_name}_{j:04d}"
        pair_dir = os.path.join(base_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)

        np.save(os.path.join(pair_dir, "mask1.npy"), mask1_raw.astype(np.float32))
        np.save(os.path.join(pair_dir, "mask2.npy"), mask2_raw.astype(np.float32))
        
        #VISUALIZATION BLOCK
        #creating a 2x2 grid: frames vs. masks
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        #frame 1
        axes[0, 0].imshow(im1)
        axes[0, 0].set_title(f"anchor ({vid1_name} f{i*stride})")
        axes[0, 0].axis('off')
        
        #mask 1
        axes[0, 1].imshow(im1)
        axes[0, 1].imshow(cv2.resize(mask1_raw, (im1.shape[1], im1.shape[0])), alpha=0.5, cmap='jet')
        axes[0, 1].set_title("joint mask 1")
        axes[0, 1].axis('off')
        
        #frame 2
        axes[1, 0].imshow(im2)
        axes[1, 0].set_title(f"positive ({vid2_name} f{j*stride})")
        axes[1, 0].axis('off')
        
        #mask 2
        axes[1, 1].imshow(im2)
        axes[1, 1].imshow(cv2.resize(mask2_raw, (im2.shape[1], im2.shape[0])), alpha=0.5, cmap='jet')
        axes[1, 1].set_title("joint mask 2")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(pair_dir, "visual_check.jpg"), dpi=150)
        plt.close(fig)
        
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
        #returns total number of cross-video pairs we generated
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
        returning the exact cross-video pair and their respective masks
        '''
        pair_path = self.pair_paths[idx]
        folder_name = os.path.basename(pair_path)
        
        #format is vid1_name_000idx_AND_vid2_name_000idx
        parts = folder_name.split("_AND_")
        
        #extracting names and indices cleanly
        vid1_parts = parts[0].split("_")
        vid1_name = "_".join(vid1_parts[:-1])
        im1_idx = int(vid1_parts[-1])
        
        vid2_parts = parts[1].split("_")
        vid2_name = "_".join(vid2_parts[:-1])
        im2_idx = int(vid2_parts[-1])
        
        #setting up paths
        pt_path1 = os.path.join(self.pt_dir, vid1_name)
        pt_path2 = os.path.join(self.pt_dir, vid2_name)
        mp4_path1 = os.path.join(self.mp4_dir, vid1_name + ".mp4")
        mp4_path2 = os.path.join(self.mp4_dir, vid2_name + ".mp4")
        
        #load flow and depth from pt dirs
        im1_flow = torch.load(os.path.join(pt_path1, "flow.pt"), weights_only=True)[im1_idx].float()
        im1_depth = torch.load(os.path.join(pt_path1, "depth.pt"), weights_only=True)[im1_idx].float()
        
        im2_flow = torch.load(os.path.join(pt_path2, "flow.pt"), weights_only=True)[im2_idx].float()
        im2_depth = torch.load(os.path.join(pt_path2, "depth.pt"), weights_only=True)[im2_idx].float()
        
        #load frames (multiplying by 2 as stride was 2)
        im1 = self._get_single_frame(mp4_path1, im1_idx * 2) 
        im2 = self._get_single_frame(mp4_path2, im2_idx * 2)
        
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
            "vid1_name": vid1_name,
            "vid2_name": vid2_name
        }        
            
if __name__ == "__main__":
    dir = "UCF_Rep/train"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #dino model
    model_name = 'facebook/dinov2-base'
    dino_model = AutoModel.from_pretrained(model_name, output_attentions=True).cuda()
    dino_model.eval()
    
    #group all videos by class before making pairs
    vid_files = [f for f in os.listdir(dir) if f.endswith('.mp4')]
    class_dict = {}
    for f in vid_files:
        cls_name = f.split('_')[1] #gets 'BreastStroke' from 'v_BreastStroke_g01_c01.mp4'
        if cls_name not in class_dict:
            class_dict[cls_name] = []
        class_dict[cls_name].append(f)
        
    #loop through classes and pick random different videos
    for cls_name, vids in class_dict.items():
        if len(vids) < 2:
            continue #need at least 2 videos to make a cross-video pair
            
        for vid1_file in vids:
            vid1_name = os.path.splitext(vid1_file)[0]
            frames1 = mp4_to_frames(os.path.join(dir, vid1_file))
            
            #pick a different random video from the same class
            valid_vids = [v for v in vids if v != vid1_file]
            if len(valid_vids) == 0:
                continue
                
            vid2_file = np.random.choice(valid_vids)
            vid2_name = os.path.splitext(vid2_file)[0]
            frames2 = mp4_to_frames(os.path.join(dir, vid2_file))
            
            #generate 3 random pairs between these two videos
            save_graph_cross_video(vid1_name, vid2_name, frames1, frames2, device, dino_model, cls_name)
            print(f"co-segmentation done for {vid1_name} and {vid2_name}")    
    