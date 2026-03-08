import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision.transforms import v2
import cv2
from ..utils.dino_exp import maskcut_tensor_method #for dino
from torchvision.models.optical_flow import raft_large #for optical flow



def mp4_to_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
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
def tensorize_vid(frames, transform):
    #to return a 4d tensor - [T, C, H, W]
    video_tensor = torch.stack([v2.functional.to_image(f) for f in frames])
    
    #the new v2 transform seems to be able to apply transforms to videos/batches..
    return transform(video_tensor)

def preprocess(vid_tensor, name, dino_model, raft_model, depth_model = None):
    '''
    input format: [f0_tensor, f1_tensor, .., fn_tensor] -> (T, C, H, W)
    output forma: save on disk like-
    
    dataset/
        ─ video_001/
            -mask0/
                - dino_mask.pt
                - flow.pt
                - depth.pt
            -mask1/
                - dino_mask.pt
                - flow.pt
                - depth.pt
        - video_002/
            - ..
    '''
    t, c, h, w = vid_tensor.shape
    vid_name = name.split('.')[0]
    base_dir = "ucfrep_intermediate_dataset"
    mask0_dir = os.path.join(base_dir, vid_name, "mask0")
    mask1_dir = os.path.join(base_dir, vid_name, "mask1")
    os.makedirs(mask0_dir, exist_ok=True)
    os.makedirs(mask1_dir, exist_ok=True)
    
    #lists to hold averaged vectors per frame
    m0_dino_list, m0_flow_list, m0_depth_list = [], [], []
    m1_dino_list, m1_flow_list, m1_depth_list = [], [], []
    
    for i in range(t):
        #get current frame and add batch dim
        frame_t = vid_tensor[i].unsqueeze(0).cuda()
        
        #1. masks from maskcut
        masks = maskcut_tensor_method(img_tensor=frame_t, num_objects=2, dino_model=dino_model) #get 2 masks
        mask0 = torch.tensor(masks[0]).cuda() # [h, w]
        mask1 = torch.tensor(masks[1]).cuda() # [h, w]
        
    

class UCFRep_train(Dataset):
    def __init__(self, vids):
        
            
        

if __name__ == "__main__":
    dir = "UCF_Rep/train"
    # train_dict = {}
    for idx, vid_file in enumerate(os.listdir(dir)):
        #each one is an mp4 file
        frames = mp4_to_frames(os.path.join(dir, vid_file)) #get frames
        # train_dict[f"video_{idx}"] = frames
        video_tensor = tensorize_vid(frames, transform)
        
        preprocess(video_tensor, vid_file)
        
    #we now have the intermediates files stored on disk
    

    

        