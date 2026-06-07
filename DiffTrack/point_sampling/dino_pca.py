import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from torchvision import transforms
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel
from sklearn.decomposition import PCA
import cv2

VIDEO_NAME  = 'v_Biking_g21_c01'
VIDEO_PATH  = f'UCF_Rep/val/{VIDEO_NAME}.mp4'
OUTPUT_PATH = 'point_sampling/dino_pca.png'

NUM_FRAMES  = 1
SIZE        = (224, 224)


def load_dino(model_name='facebook/dinov2-base' , device = 'cuda'):
    '''
    getting a tensor of shape- [B, 37, 37, 768] 
    '''
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    return model, device

def get_dino_feats(model, images, device):
    if not isinstance(images, list):
        images = [images]

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    batch = torch.stack([preprocess(img) for img in images]).to(device)

    with torch.no_grad():
        outputs = model(batch)
        
    patch_tokens = outputs.last_hidden_state[:, 1:, :] #skips cls token, shape: [B, 256, 768] as
    #it is not yet a spatial gerid of 16x16 yet.
    return patch_tokens


def perform_pca(dino_embeds, image): #image is neede for overlay purposes
    #take [B, 256, 768] and turn it into [B, 16, 16, 3]
    B, num_patches, dim = dino_embeds.shape
    flat = dino_embeds.cpu().numpy().reshape(-1, dim)  # (B*256, 768)  
    
    pca = PCA(n_components=3)
    projected = pca.fit_transform(flat) #now its (b*256, 3)
    
    projected = (projected - projected.min(axis=0)) / (projected.max(axis=0) - projected.min(axis=0) + 1e-8) #normalization
    projected = projected.reshape(B, 16, 16, 3) #spatialization, so upsamping can be done later
    
    return projected, pca


if __name__ == "__main__":
    #load image, pass to dino
    model, device = load_dino("facebook/dinov2-base")

    cap     = cv2.VideoCapture(VIDEO_PATH)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

    images = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        assert ret, f"failed to read frame {idx} from {VIDEO_PATH}"
        images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    patch_tokens = get_dino_feats(model, images, device)  # [NUM_FRAMES, 256, 768]
    pca_features, _ = perform_pca(patch_tokens, images[0])  # [NUM_FRAMES, 16, 16, 3]

    fig, axes = plt.subplots(2, NUM_FRAMES, figsize=(3 * NUM_FRAMES, 6))
    if NUM_FRAMES == 1:
        axes = axes[:, np.newaxis]

    for t in range(NUM_FRAMES):
        axes[0, t].imshow(images[t])
        axes[0, t].set_title(f'frame {t}')
        axes[0, t].axis('off')

        pca_img = (pca_features[t] * 255).astype(np.uint8)
        pca_upsampled = np.array(Image.fromarray(pca_img).resize(images[t].size, Image.NEAREST))
        axes[1, t].imshow(pca_upsampled)
        axes[1, t].set_title(f'dino pca {t}')
        axes[1, t].axis('off')

    plt.suptitle(f'{VIDEO_NAME}  |  DINO (PCA->RGB, shared across frames)')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")

    
    
    


    