import os
import sys
import pickle
from pathlib import Path

import cvxpy as cp
import imageio.v3 as iio
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from PIL import Image, ImageDraw


# ============================================================
# setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

vid_1_info = "point_track/saved_videos/v_BodyWeightSquats_g21_c04/v_BodyWeightSquats_g21_c04_trajectories_frame_22.pkl"
vid_2_info = "point_track/saved_videos/v_BodyWeightSquats_g22_c03/v_BodyWeightSquats_g22_c03_trajectories_frame_22.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_1_PATH = "UCF_Rep/val/v_BodyWeightSquats_g21_c04.mp4"
VIDEO_2_PATH = "UCF_Rep/val/v_BodyWeightSquats_g22_c03.mp4"

ANCHOR_FRAME = 0              # using first frames, as you requested
IMG_SIZE = 224                # keep things manageable
PATCH_SIZE = 14               # DINOv2 ViT-B/14
EIGEN_NUM = 30               # spectral basis size
COPCA_DIM = 128              # pairwise joint PCA dimension
TOPK_PER_POINT = 1           # final match per source point
VIS_THRESHOLD = 0.5          # use visible points at anchor frame
SAVE_DIR = Path("point_track/fmap_matches")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# simple utilities
# ============================================================


def point_xy_to_patch_index(points_xy, img_hw, patch_hw):
    """
    points_xy: (N,2) in resized image pixel coords
    img_hw: (H,W), e.g. (224,224)
    patch_hw: (Hp,Wp), e.g. (16,16)

    returns:
        patch_idx: (N,)
        patch_xy:  (N,2) integer patch coords
    """
    H, W = img_hw
    Hp, Wp = patch_hw

    pts = points_xy.clone().float()
    px = torch.clamp((pts[:, 0] / W) * Wp, 0, Wp - 1e-6).long()
    py = torch.clamp((pts[:, 1] / H) * Hp, 0, Hp - 1e-6).long()

    patch_idx = py * Wp + px
    patch_xy = torch.stack([px, py], dim=1)
    return patch_idx, patch_xy


def patch_index_to_patch_center_xy(patch_idx, patch_hw, img_hw, device=None):
    """
    patch_idx: (N,)
    patch_hw: (Hp,Wp)
    img_hw: (H,W)

    returns pixel centers in resized image coords: (N,2)
    """
    Hp, Wp = patch_hw
    H, W = img_hw

    patch_idx = patch_idx.long()
    py = patch_idx // Wp
    px = patch_idx % Wp

    cell_w = W / Wp
    cell_h = H / Hp

    x = (px.float() + 0.5) * cell_w
    y = (py.float() + 0.5) * cell_h

    out = torch.stack([x, y], dim=1)
    if device is not None:
        out = out.to(device)
    return out

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_video_frame(video_path, frame_idx):
    frames = iio.imread(video_path, plugin="FFMPEG")
    if frame_idx < 0 or frame_idx >= len(frames):
        raise IndexError(f"frame {frame_idx} out of bounds for {video_path}")
    return frames[frame_idx]


def resize_and_normalize(frame_np, img_size=224):
    """
    returns:
        pil_img_resized
        tensor_norm : (1,3,H,W)
    """
    pil_img = Image.fromarray(frame_np).convert("RGB")
    pil_img = pil_img.resize((img_size, img_size), Image.Resampling.BILINEAR)

    arr = np.array(pil_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    tensor = (tensor - mean) / std
    return pil_img, tensor.to(device)


def get_visible_points_from_tracks(track_info, anchor_frame=0, vis_threshold=0.5):
    """
    expects:
      tracks: (1, T, N, 2)
      visibility: (1, T, N) or similar
    returns:
      points_xy: (N_vis, 2) in original video pixel coords
      selected_track_ids: indices into original point set
    """
    tracks = torch.tensor(track_info["tracks"], dtype=torch.float32)
    visibility = torch.tensor(track_info["visibility"], dtype=torch.float32)

    pts = tracks[0, anchor_frame]  # (N,2)

    if visibility.ndim == 3:
        vis = visibility[0, anchor_frame] > vis_threshold
    elif visibility.ndim == 4:
        vis = visibility[0, anchor_frame, :, 0] > vis_threshold
    else:
        vis = torch.ones(pts.shape[0], dtype=torch.bool)

    selected_ids = torch.nonzero(vis, as_tuple=False).squeeze(1)
    pts = pts[selected_ids]

    return pts, selected_ids


def rescale_points(points_xy, orig_hw, new_hw):
    """
    points_xy in original image coords -> resized image coords
    """
    orig_h, orig_w = orig_hw
    new_h, new_w = new_hw

    pts = points_xy.clone().float()
    pts[:, 0] = pts[:, 0] * (new_w / orig_w)
    pts[:, 1] = pts[:, 1] * (new_h / orig_h)
    return pts


# ============================================================
# DINOv2 dense descriptor extraction
# ============================================================
def extract_dense_dino_features(model, img_tensor, patch_size=14):
    """
    img_tensor: (1,3,H,W)
    returns:
        feat_map: (Hp, Wp, C)
    """
    with torch.no_grad():
        out = model.forward_features(img_tensor)
        patch_tokens = out["x_norm_patchtokens"]  # (1, N, C)

    _, n, c = patch_tokens.shape
    h, w = img_tensor.shape[-2:]
    hp, wp = h // patch_size, w // patch_size

    if hp * wp != n:
        raise ValueError(f"patch mismatch: got {n}, expected {hp*wp}")

    feat_map = patch_tokens[0].reshape(hp, wp, c).contiguous()
    return feat_map


# ============================================================
# their co-PCA idea
# ============================================================
def co_pca_pair(feat1_hwc, feat2_hwc, out_dim=128):
    """
    faithful adaptation of their pairwise co-PCA.
    inputs:
        feat1_hwc: (H1,W1,C)
        feat2_hwc: (H2,W2,C)
    returns:
        proj1: (H1,W1,out_dim)
        proj2: (H2,W2,out_dim)
    """
    h1, w1, c = feat1_hwc.shape
    h2, w2, _ = feat2_hwc.shape

    x1 = feat1_hwc.reshape(-1, c)   # (N1,C)
    x2 = feat2_hwc.reshape(-1, c)   # (N2,C)

    x = torch.cat([x1, x2], dim=0)
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean

    q = min(out_dim, x_centered.shape[1], x_centered.shape[0] - 1)
    U, S, V = torch.pca_lowrank(x_centered, q=q)
    x_reduced = x_centered @ V[:, :q]

    x1_red = x_reduced[: x1.shape[0]].reshape(h1, w1, q)
    x2_red = x_reduced[x1.shape[0] :].reshape(h2, w2, q)

    return x1_red.contiguous(), x2_red.contiguous()


# ============================================================
# graph + laplacian, following their repo closely
# ============================================================
def check_and_derive_sigma(sigma, values):
    if sigma == "median":
        return 2 * (torch.median(values) ** 2)
    else:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        return sigma


class ImageGraph:
    def __init__(self, feat_hwc, use_dino=True):
        """
        feat_hwc: (H,W,C)
        builds a 2d grid graph like their repo
        """
        self.device = feat_hwc.device
        self.feat_hwc = feat_hwc
        h, w, _ = feat_hwc.shape

        grid_graph = nx.grid_2d_graph(h, w)
        grid_graph = nx.convert_node_labels_to_integers(grid_graph)
        edge_index = torch.tensor(list(grid_graph.edges()), device=self.device).t().contiguous()

        self.graph = torch_geometric.data.Data(
            edge_index=edge_index,
            num_nodes=h * w,
        )
        self.adjust_weights_via_feature_differences(feat_hwc, use_dino=use_dino)
        
    def adjust_weights_via_feature_differences(self, feat_hwc, use_dino=True):
        h, w, c = feat_hwc.shape
        num_nodes = h * w

        edge_index = self.graph.edge_index.to(feat_hwc.device)

        if use_dino:
            feats = feat_hwc.reshape(num_nodes, c)
            f_from = feats[edge_index[0]]
            f_to = feats[edge_index[1]]

            feat_values = torch.norm(f_from - f_to, dim=1)
            feat_val_offset = feat_values - feat_values.mean()
            feat_val_offset = feat_val_offset / 5.0
            feat_values = feat_values.mean() + feat_val_offset
        else:
            feat_values = torch.ones((edge_index.shape[1],), device=feat_hwc.device)

        spatial_values = torch.ones((edge_index.shape[1],), device=feat_hwc.device)

        sigma_f = check_and_derive_sigma("median", feat_values)
        sigma_s = check_and_derive_sigma("median", spatial_values)

        feat_values = torch.exp(-torch.square(feat_values) / sigma_f)
        spatial_values = torch.exp(-torch.square(spatial_values) / sigma_s)

        edge_attr = feat_values * spatial_values
        self.graph.edge_attr = edge_attr


class Laplacian:
    def __init__(self, graph_data):
        self.graph = graph_data
        self.num_nodes = graph_data.num_nodes
        self.laplacian = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.compute_laplacian()

    def compute_laplacian(self):
        e_i = self.graph.edge_index
        e_w = self.graph.edge_attr

        e_i_undirected = torch.cat([e_i, e_i.flip(dims=[0])], dim=1)
        e_w_undirected = torch.cat([e_w, e_w], dim=0)

        self.laplacian = torch_geometric.utils.get_laplacian(
            e_i_undirected,
            e_w_undirected,
            normalization="sym",
        )

    def compute_spectra(self, eigen_num):
        idx, vals = self.laplacian
        sparse = torch.sparse_coo_tensor(idx, vals, size=(self.num_nodes, self.num_nodes)).coalesce()

        eigvals, eigvecs = torch.lobpcg(sparse, k=eigen_num, largest=False)
        eigvecs = torch.linalg.qr(eigvecs)[0]
        eigvecs = F.normalize(eigvecs, p=2, dim=0)

        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs
        return eigvals, eigvecs

    def project_functions(self, eigen_num, functions):
        """
        functions: (num_nodes, dim)
        """
        if self.eigenvectors is None:
            self.compute_spectra(eigen_num)

        return self.eigenvectors[:, :eigen_num].T @ functions


def solve_functional_map_cvx(proj_src, proj_tgt):
    """
    faithful to their cvx solve idea:
      minimize || F @ proj_src - proj_tgt ||_F
      subject to F >= 0

    proj_src: (K, D)
    proj_tgt: (K, D)

    returns:
      F_map: (K,K)
    """
    k = proj_src.shape[0]

    F_var = cp.Variable((k, k))
    src_np = proj_src.detach().cpu().numpy()
    tgt_np = proj_tgt.detach().cpu().numpy()

    residual = cp.norm(F_var @ src_np - tgt_np, "fro")
    problem = cp.Problem(cp.Minimize(residual), [F_var >= 0])
    problem.solve()

    if F_var.value is None:
        raise RuntimeError("cvxpy failed to solve functional map")

    F_map = torch.tensor(F_var.value, dtype=torch.float32, device=proj_src.device)
    return F_map


class FunctionalMap:
    def __init__(self, src_data_hwc, tgt_data_hwc, eigen_num=30):
        """
        src_data_hwc, tgt_data_hwc: (H,W,D)
        """
        self.src_data = src_data_hwc
        self.tgt_data = tgt_data_hwc
        self.eigen_num = eigen_num
        self.dim = src_data_hwc.shape[-1]

        self.src_h, self.src_w, _ = src_data_hwc.shape
        self.tgt_h, self.tgt_w, _ = tgt_data_hwc.shape

        self.src_graph = ImageGraph(src_data_hwc, use_dino=True)
        self.tgt_graph = ImageGraph(tgt_data_hwc, use_dino=True)

        self.src_lap = Laplacian(self.src_graph.graph)
        self.tgt_lap = Laplacian(self.tgt_graph.graph)

        self.src_eigvals, self.src_basis = self.src_lap.compute_spectra(eigen_num)   # (Ns, K)
        self.tgt_eigvals, self.tgt_basis = self.tgt_lap.compute_spectra(eigen_num)   # (Nt, K)

        self.transition_matrix = self.compute_transition_matrix()                     # (K, K)

    def compute_transition_matrix(self):
        src_funcs = self.src_data.reshape(-1, self.dim)   # (Ns, D)
        tgt_funcs = self.tgt_data.reshape(-1, self.dim)   # (Nt, D)

        proj_src = self.src_lap.project_functions(self.eigen_num, src_funcs)  # (K, D)
        proj_tgt = self.tgt_lap.project_functions(self.eigen_num, tgt_funcs)  # (K, D)

        F_map = solve_functional_map_cvx(proj_src, proj_tgt)                   # (K, K)
        return F_map

    def get_pointwise_map(self):
        """
        Convert functional map to a pointwise map from source patches to target patches.

        Standard idea:
            embed_src = Phi_src @ C^T
            embed_tgt = Phi_tgt

        Then NN from each source embedding row to target embedding row.
        """
        Phi_s = self.src_basis[:, : self.eigen_num]   # (Ns, K)
        Phi_t = self.tgt_basis[:, : self.eigen_num]   # (Nt, K)
        C = self.transition_matrix                    # (K, K)

        embed_src = Phi_s @ C.T                       # (Ns, K)
        embed_tgt = Phi_t                             # (Nt, K)

        embed_src = F.normalize(embed_src, dim=1)
        embed_tgt = F.normalize(embed_tgt, dim=1)

        sim = embed_src @ embed_tgt.T                # (Ns, Nt)
        tgt_idx = sim.argmax(dim=1)                  # (Ns,)

        return tgt_idx, sim


# ============================================================
# point-level descriptor sampling
# ============================================================
def sample_dense_descriptor_at_points(feat_hwc, pts_xy, img_hw):
    """
    feat_hwc: (Hf,Wf,C)
    pts_xy: (N,2) in resized image coordinates
    img_hw: resized image size, e.g. (224,224)

    returns:
      desc: (N,C)
    """
    h_img, w_img = img_hw
    h_feat, w_feat, c = feat_hwc.shape

    feat = feat_hwc.permute(2, 0, 1).unsqueeze(0)  # (1,C,Hf,Wf)

    x = pts_xy[:, 0] / max(w_img - 1, 1) * 2 - 1
    y = pts_xy[:, 1] / max(h_img - 1, 1) * 2 - 1
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2).to(feat_hwc.device)

    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1,C,N,1)

    sampled = sampled.squeeze(0).squeeze(-1).T.contiguous()  # (N,C)
    return sampled


# ============================================================
# matching logic
# ============================================================
def match_points_via_functional_map(
    fmap_model,
    src_pts_xy,
    tgt_pts_xy,
    img_hw,
):
    """
    Use the dense pointwise patch map induced by the functional map.

    Steps:
      1. map each source tracked point -> nearest source patch
      2. source patch -> target patch via functional map
      3. target patch center -> nearest target tracked point
    """
    src_patch_hw = (fmap_model.src_h, fmap_model.src_w)
    tgt_patch_hw = (fmap_model.tgt_h, fmap_model.tgt_w)

    # dense source-patch -> target-patch mapping
    pointwise_tgt_idx, sim = fmap_model.get_pointwise_map()   # (Ns_patches,), (Ns_patches, Nt_patches)

    # map source tracked points to source patches
    src_patch_idx, _ = point_xy_to_patch_index(src_pts_xy, img_hw, src_patch_hw)  # (Ns_points,)

    # get mapped target patch for each source tracked point
    mapped_tgt_patch_idx = pointwise_tgt_idx[src_patch_idx]   # (Ns_points,)

    # convert mapped target patches to target image pixel centers
    mapped_tgt_patch_centers = patch_index_to_patch_center_xy(
        mapped_tgt_patch_idx,
        tgt_patch_hw,
        img_hw,
        device=tgt_pts_xy.device,
    )   # (Ns_points, 2)

    matches = []

    # snap mapped target patch center to nearest target tracked point
    for i in range(src_pts_xy.shape[0]):
        pred_xy = mapped_tgt_patch_centers[i : i + 1]         # (1,2)
        dists = torch.cdist(pred_xy, tgt_pts_xy)              # (1,Nt_points)
        j = torch.argmin(dists, dim=1).item()
        dist = dists[0, j].item()

        matches.append(
            {
                "src_idx": int(i),
                "tgt_idx": int(j),
                "patch_dist": float(dist),
                "src_patch_idx": int(src_patch_idx[i].item()),
                "mapped_tgt_patch_idx": int(mapped_tgt_patch_idx[i].item()),
            }
        )

    return matches

# ============================================================
# visualization
# ============================================================
def draw_matches(img1_pil, img2_pil, src_pts, tgt_pts, matches, save_path):
    img1 = img1_pil.copy()
    img2 = img2_pil.copy()

    w1, h1 = img1.size
    w2, h2 = img2.size

    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), (255, 255, 255))
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (w1, 0))

    draw = ImageDraw.Draw(canvas)

    for m in matches:
        i = m["src_idx"]
        j = m["tgt_idx"]

        x1, y1 = src_pts[i].tolist()
        x2, y2 = tgt_pts[j].tolist()
        x2_shift = x2 + w1

        r = 3
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=(255, 0, 0))
        draw.ellipse((x2_shift - r, y2 - r, x2_shift + r, y2 + r), fill=(0, 0, 255))
        draw.line((x1, y1, x2_shift, y2), fill=(0, 200, 0), width=1)

    canvas.save(save_path)


# ============================================================
# main
# ============================================================
def main():
    print("loading trajectory info...")
    info1 = load_pickle(vid_1_info)
    info2 = load_pickle(vid_2_info)

    print("loading anchor frames...")
    frame1_np = load_video_frame(VIDEO_1_PATH, ANCHOR_FRAME)
    frame2_np = load_video_frame(VIDEO_2_PATH, ANCHOR_FRAME)

    orig_hw1 = frame1_np.shape[:2]
    orig_hw2 = frame2_np.shape[:2]

    img1_pil, img1_tensor = resize_and_normalize(frame1_np, IMG_SIZE)
    img2_pil, img2_tensor = resize_and_normalize(frame2_np, IMG_SIZE)

    print("collecting tracked points visible at anchor frame...")
    src_pts_orig, src_ids = get_visible_points_from_tracks(info1, anchor_frame=ANCHOR_FRAME, vis_threshold=VIS_THRESHOLD)
    tgt_pts_orig, tgt_ids = get_visible_points_from_tracks(info2, anchor_frame=ANCHOR_FRAME, vis_threshold=VIS_THRESHOLD)

    src_pts = rescale_points(src_pts_orig, orig_hw1, (IMG_SIZE, IMG_SIZE)).to(device)
    tgt_pts = rescale_points(tgt_pts_orig, orig_hw2, (IMG_SIZE, IMG_SIZE)).to(device)

    print(f"source visible points: {src_pts.shape[0]}")
    print(f"target visible points: {tgt_pts.shape[0]}")

    print("loading DINOv2...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()

    print("extracting dense DINO features...")
    feat1 = extract_dense_dino_features(dino, img1_tensor, PATCH_SIZE)   # (Hf,Wf,C)
    feat2 = extract_dense_dino_features(dino, img2_tensor, PATCH_SIZE)

    print("running pairwise co-PCA...")
    feat1_copca, feat2_copca = co_pca_pair(feat1, feat2, out_dim=COPCA_DIM)  # (Hf,Wf,D)

    print("building functional map...")
    fmap_model = FunctionalMap(
        src_data_hwc=feat1_copca,
        tgt_data_hwc=feat2_copca,
        eigen_num=EIGEN_NUM,
    )

    print("matching source tracked points to target tracked points via dense FMAP patch map...")
    matches = match_points_via_functional_map(
        fmap_model=fmap_model,
        src_pts_xy=src_pts,
        tgt_pts_xy=tgt_pts,
        img_hw=(IMG_SIZE, IMG_SIZE),
    )

    # keep best unique match per source point
    best_by_src = {}
    for m in matches:
        i = m["src_idx"]
        if i not in best_by_src or m["patch_dist"] > best_by_src[i]["patch_dist"]:
            best_by_src[i] = m

    final_matches = [best_by_src[i] for i in sorted(best_by_src.keys())]

    print(f"final matches: {len(final_matches)}")

    out_data = {
        "video1": VIDEO_1_PATH,
        "video2": VIDEO_2_PATH,
        "anchor_frame": ANCHOR_FRAME,
        "source_track_ids": src_ids.cpu().numpy(),
        "target_track_ids": tgt_ids.cpu().numpy(),
        "source_points_resized": src_pts.detach().cpu().numpy(),
        "target_points_resized": tgt_pts.detach().cpu().numpy(),
        "matches": final_matches,
        "eigen_num": EIGEN_NUM,
        "copca_dim": COPCA_DIM,
        "img_size": IMG_SIZE,
    }

    out_pkl = SAVE_DIR / "fmap_point_matches.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(out_data, f)
    print(f"saved matches to {out_pkl}")

    out_png = SAVE_DIR / "fmap_point_matches.png"
    draw_matches(
        img1_pil,
        img2_pil,
        src_pts.detach().cpu(),
        tgt_pts.detach().cpu(),
        final_matches,
        out_png,
    )
    print(f"saved visualization to {out_png}")


if __name__ == "__main__":
    main()