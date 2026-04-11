import os
import sys
import pickle
from pathlib import Path
from collections import OrderedDict

import cvxpy as cp
import imageio.v3 as iio
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from matplotlib import cm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2

GPU_ID = 1
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)

# setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COTRACKER_ROOT = Path(__file__).resolve().parent / "co-tracker"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.utils.visualizer import Visualizer
from point_sampling.model_finetune import DINOv2_LoRA
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# hardcoded paths
# ------------------------------------------------------------
video_coach_path = "UCF_Rep/val/v_PlayingCello_g22_c01.mp4"
video_student_path = "UCF_Rep/val/v_PlayingCello_g23_c06.mp4"

if not os.path.isfile(video_coach_path) or not os.path.isfile(video_student_path):
    raise FileNotFoundError("one or both videos not found")

coach_name = Path(video_coach_path).stem
student_name = Path(video_student_path).stem
out_dir_name = f"compare_{coach_name}_vs_{student_name}"
save_dir = os.path.join("point_track/saved_videos", out_dir_name)
os.makedirs(save_dir, exist_ok=True)

# ------------------------------------------------------------
# config
# ------------------------------------------------------------
IMG_SIZE = 224
PATCH_GRID = 16
COPCA_DIM = 128
EIGEN_NUM = 30

start_f_coach = 20
start_f_student = 20
mask_frame_idx = 1

query_frame_coach = start_f_coach + (mask_frame_idx * 2)
query_frame_student = start_f_student + (mask_frame_idx * 2)

mask_threshold = 0.15
max_query_points = 30
pre_fps_pool_size = 1000
target_radius = 0
mutual_only = False

raft_transform = v2.Compose([
    v2.ConvertImageDtype(torch.float32),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    v2.Resize(size=(520, 960)),
])

# ------------------------------------------------------------
# video load
# ------------------------------------------------------------
frames_coach = iio.imread(video_coach_path, plugin="FFMPEG")
meta_coach = iio.immeta(video_coach_path, plugin="FFMPEG")
fps_coach = float(meta_coach.get("fps", 30))

frames_student = iio.imread(video_student_path, plugin="FFMPEG")
meta_student = iio.immeta(video_student_path, plugin="FFMPEG")
fps_student = float(meta_student.get("fps", 30))

video_tensor_coach = torch.tensor(frames_coach).permute(0, 3, 1, 2)[None].float().to(device)
video_tensor_student = torch.tensor(frames_student).permute(0, 3, 1, 2)[None].float().to(device)

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def extract_plain_dino_patch_features(dino_model, pixel_frames, frame_idx=1, patch_grid=16):
    img = pixel_frames[frame_idx:frame_idx + 1]
    with torch.no_grad():
        out = dino_model.forward_features(img)
        patch_tokens = out["x_norm_patchtokens"]
    _, n, c = patch_tokens.shape
    assert n == patch_grid * patch_grid, f"expected {patch_grid*patch_grid} patches, got {n}"
    feat_hwc = patch_tokens[0].reshape(patch_grid, patch_grid, c).contiguous()
    return feat_hwc


def build_three_frame_chunk(frames_np, start_frame=20, stride=2, num_frames=3, target_size=(224, 224)):
    idxs = [start_frame + i * stride for i in range(num_frames)]
    if idxs[-1] >= len(frames_np):
        raise ValueError("not enough frames for sampling")

    sampled = torch.from_numpy(frames_np[idxs]).permute(0, 3, 1, 2).float() / 255.0
    sampled = F.interpolate(sampled, size=target_size, mode="bilinear", align_corners=False)

    raw_frames = sampled.permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().cpu().numpy()

    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    pixel_frames = (sampled.to(device) - norm_mean) / norm_std

    return raw_frames, pixel_frames


def farthest_point_sampling_2d(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    if points.shape[0] <= num_samples:
        return points

    points_float = points.float()
    selected_indices = torch.empty(num_samples, dtype=torch.long, device=points.device)

    center = points_float.mean(dim=0, keepdim=True)
    farthest_index = torch.argmax(torch.sum((points_float - center) ** 2, dim=1))
    min_distances = torch.full((points.shape[0],), float("inf"), device=points.device)

    for sample_index in range(num_samples):
        selected_indices[sample_index] = farthest_index
        selected_point = points_float[farthest_index:farthest_index + 1]
        distances = torch.sum((points_float - selected_point) ** 2, dim=1)
        min_distances = torch.minimum(min_distances, distances)
        farthest_index = torch.argmax(min_distances)

    return points[selected_indices]


def co_pca_pair(feat1_hwc, feat2_hwc, out_dim=128):
    h1, w1, c = feat1_hwc.shape
    h2, w2, _ = feat2_hwc.shape

    x1 = feat1_hwc.reshape(-1, c)
    x2 = feat2_hwc.reshape(-1, c)

    x = torch.cat([x1, x2], dim=0)
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean

    q = min(out_dim, x_centered.shape[1], x_centered.shape[0] - 1)
    U, S, V = torch.pca_lowrank(x_centered, q=q)
    x_reduced = x_centered @ V[:, :q]

    x1_red = x_reduced[:x1.shape[0]].reshape(h1, w1, q)
    x2_red = x_reduced[x1.shape[0]:].reshape(h2, w2, q)
    return x1_red.contiguous(), x2_red.contiguous()


def check_and_derive_sigma(sigma, values):
    if sigma == "median":
        return 2 * (torch.median(values) ** 2)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return sigma


class ImageGraph:
    def __init__(self, feat_hwc, use_dino=True):
        self.device = feat_hwc.device
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
        self.graph.edge_attr = feat_values * spatial_values


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
        if self.eigenvectors is None:
            self.compute_spectra(eigen_num)
        return self.eigenvectors[:, :eigen_num].T @ functions


def solve_functional_map_cvx(proj_src, proj_tgt):
    k = proj_src.shape[0]
    F_var = cp.Variable((k, k))
    src_np = proj_src.detach().cpu().numpy()
    tgt_np = proj_tgt.detach().cpu().numpy()
    residual = cp.norm(F_var @ src_np - tgt_np, "fro")
    problem = cp.Problem(cp.Minimize(residual), [F_var >= 0])
    problem.solve()

    if F_var.value is None:
        raise RuntimeError("cvxpy failed to solve functional map")

    return torch.tensor(F_var.value, dtype=torch.float32, device=proj_src.device)


class FunctionalMap:
    def __init__(self, src_data_hwc, tgt_data_hwc, eigen_num=30):
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

        self.src_eigvals, self.src_basis = self.src_lap.compute_spectra(eigen_num)
        self.tgt_eigvals, self.tgt_basis = self.tgt_lap.compute_spectra(eigen_num)

        self.transition_matrix = self.compute_transition_matrix()

    def compute_transition_matrix(self):
        src_funcs = self.src_data.reshape(-1, self.dim)
        tgt_funcs = self.tgt_data.reshape(-1, self.dim)

        proj_src = self.src_lap.project_functions(self.eigen_num, src_funcs)
        proj_tgt = self.tgt_lap.project_functions(self.eigen_num, tgt_funcs)

        return solve_functional_map_cvx(proj_src, proj_tgt)

    def get_pointwise_map(self):
        phi_s = self.src_basis[:, :self.eigen_num]
        phi_t = self.tgt_basis[:, :self.eigen_num]
        C = self.transition_matrix

        embed_src = phi_s @ C.T
        embed_tgt = phi_t

        embed_src = F.normalize(embed_src, dim=1)
        embed_tgt = F.normalize(embed_tgt, dim=1)

        sim = embed_src @ embed_tgt.T
        best_tgt = sim.argmax(dim=1)
        best_score = sim.max(dim=1).values

        top2_vals, _ = torch.topk(sim, k=min(2, sim.shape[1]), dim=1)
        margin = top2_vals[:, 0] - top2_vals[:, 1] if sim.shape[1] > 1 else top2_vals[:, 0]

        return best_tgt, best_score, margin, sim

    def get_reverse_pointwise_map(self):
        phi_s = self.src_basis[:, :self.eigen_num]
        phi_t = self.tgt_basis[:, :self.eigen_num]
        C = self.transition_matrix

        embed_src = phi_s
        embed_tgt = phi_t @ C

        embed_src = F.normalize(embed_src, dim=1)
        embed_tgt = F.normalize(embed_tgt, dim=1)

        sim = embed_tgt @ embed_src.T
        best_src = sim.argmax(dim=1)
        best_score = sim.max(dim=1).values

        return best_src, best_score, sim


def patch_index_to_center_xy(patch_idx, patch_hw=(16, 16), img_hw=(224, 224), device=None):
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


def save_mask_overlay(raw_frame, mask_map, out_path):
    frame = raw_frame.astype("float32")
    mask_color = (cm.get_cmap("jet")(mask_map)[..., :3] * 255.0).astype("float32")
    alpha = mask_map[..., None]
    overlay = (frame * (1.0 - alpha) + mask_color * alpha).clip(0, 255).astype("uint8")
    iio.imwrite(out_path, overlay)


def save_pair_overlay(img1, img2, pts1, pts2, out_path):
    pil1 = Image.fromarray(img1)
    pil2 = Image.fromarray(img2)

    w1, h1 = pil1.size
    w2, h2 = pil2.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), (255, 255, 255))
    canvas.paste(pil1, (0, 0))
    canvas.paste(pil2, (w1, 0))

    draw = ImageDraw.Draw(canvas)

    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i].tolist()
        x2, y2 = pts2[i].tolist()
        x2s = x2 + w1

        r = 3
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=(255, 0, 0))
        draw.ellipse((x2s - r, y2 - r, x2s + r, y2 + r), fill=(0, 0, 255))
        draw.line((x1, y1, x2s, y2), fill=(0, 255, 0), width=1)

    canvas.save(out_path)


def build_smooth_rainbow_image(height, width):
    ys = np.linspace(-1.0, 1.0, height)
    xs = np.linspace(-1.0, 1.0, width)
    xx, yy = np.meshgrid(xs, ys)

    angle = np.arctan2(yy, xx) / (2 * np.pi) + 0.5
    radius = np.sqrt(xx**2 + yy**2)
    radius = np.clip(radius, 0.0, 1.0)

    hsv = np.zeros((height, width, 3), dtype=np.float32)
    hsv[..., 0] = angle
    hsv[..., 1] = 0.25 + 0.75 * radius
    hsv[..., 2] = 1.0

    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return rgb


def image_to_patch_colors(rgb_image, patch_grid=16):
    H, W, _ = rgb_image.shape
    ph = H // patch_grid
    pw = W // patch_grid

    patch_colors = []
    for py in range(patch_grid):
        for px in range(patch_grid):
            y0 = py * ph
            y1 = (py + 1) * ph
            x0 = px * pw
            x1 = (px + 1) * pw
            patch = rgb_image[y0:y1, x0:x1]
            patch_colors.append(patch.mean(axis=(0, 1)))

    patch_colors = np.stack(patch_colors, axis=0)
    return patch_colors


def patch_colors_to_image(patch_colors, patch_grid=16, out_size=224, mode="bilinear"):
    patch_map = torch.tensor(
        patch_colors.reshape(patch_grid, patch_grid, 3),
        dtype=torch.float32,
        device=device,
    )
    patch_map = patch_map.permute(2, 0, 1).unsqueeze(0)
    patch_map = F.interpolate(
        patch_map,
        size=(out_size, out_size),
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None,
    )
    patch_map = patch_map.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    patch_map = np.clip(patch_map, 0.0, 1.0)
    patch_map = (patch_map * 255).astype(np.uint8)
    return patch_map


def save_smooth_fullimage_correspondence_plot(
    raw_source_frame,
    raw_target_frame,
    src_to_tgt,
    out_path,
    patch_grid=16,
    upsample_to=224,
):
    H, W, _ = raw_target_frame.shape
    N = patch_grid * patch_grid

    target_rainbow_full = build_smooth_rainbow_image(H, W)
    tgt_patch_colors = image_to_patch_colors(target_rainbow_full, patch_grid=patch_grid)

    src_patch_colors = np.ones((N, 3), dtype=np.float32)
    for s_idx in range(N):
        t_idx = int(src_to_tgt[s_idx].item())
        src_patch_colors[s_idx] = tgt_patch_colors[t_idx]

    tgt_color_sum = np.zeros((N, 3), dtype=np.float32)
    tgt_color_count = np.zeros((N, 1), dtype=np.float32)

    for s_idx in range(N):
        t_idx = int(src_to_tgt[s_idx].item())
        tgt_color_sum[t_idx] += src_patch_colors[s_idx]
        tgt_color_count[t_idx] += 1.0

    tgt_patch_colors_mapped = np.ones((N, 3), dtype=np.float32)
    valid = tgt_color_count.squeeze(-1) > 0
    tgt_patch_colors_mapped[valid] = tgt_color_sum[valid] / tgt_color_count[valid]

    src_vis = patch_colors_to_image(
        src_patch_colors,
        patch_grid=patch_grid,
        out_size=upsample_to,
        mode="bilinear",
    )
    tgt_vis = patch_colors_to_image(
        tgt_patch_colors_mapped,
        patch_grid=patch_grid,
        out_size=upsample_to,
        mode="bilinear",
    )

    raw_source = raw_source_frame.astype(np.uint8)
    raw_target = raw_target_frame.astype(np.uint8)

    canvas = np.ones((upsample_to, upsample_to * 4, 3), dtype=np.uint8) * 255
    canvas[:, 0:upsample_to] = raw_source
    canvas[:, upsample_to:2*upsample_to] = src_vis
    canvas[:, 2*upsample_to:3*upsample_to] = raw_target
    canvas[:, 3*upsample_to:4*upsample_to] = tgt_vis

    iio.imwrite(out_path, canvas)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
print("loading mask model, plain DINO, and RAFT...")
mask_model = DINOv2_LoRA().to(device)
model_path = "test_models/physics_guide_lora_dino_epoch_9.pth"
state_dict = torch.load(model_path, map_location=device, weights_only=True)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v

mask_model.load_state_dict(new_state_dict)
mask_model.eval()

plain_dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
raft_model = raft_large(pretrained=True, progress=False).to(device).eval()

raw_coach, px_coach = build_three_frame_chunk(frames_coach, start_frame=start_f_coach)
raw_student, px_student = build_three_frame_chunk(frames_student, start_frame=start_f_student)

print("generating optical flow for coach template...")
flow_list = []
for i in range(2):
    img1 = raft_transform(torch.from_numpy(raw_coach[i]).permute(2, 0, 1)).to(device).unsqueeze(0)
    img2 = raft_transform(torch.from_numpy(raw_coach[i + 1]).permute(2, 0, 1)).to(device).unsqueeze(0)
    with torch.no_grad():
        list_of_flows = raft_model(img1, img2)
        flow_res = F.interpolate(list_of_flows[-1], size=(224, 224), mode="bilinear", align_corners=False)
        flow_list.append(flow_res.squeeze(0))

flow_list.append(flow_list[-1].clone())
flow_tensor = torch.stack(flow_list)

print("running mask-model inference...")
with torch.no_grad():
    out_coach = mask_model(px_coach.unsqueeze(0))
    out_student = mask_model(px_student.unsqueeze(0))

pred_mask_coach = out_coach["pred_mask"]
pred_mask_student = out_student["pred_mask"]

pred_mask_coach_224 = F.interpolate(
    pred_mask_coach[0], size=(224, 224), mode="bilinear", align_corners=False
)
pred_mask_student_224 = F.interpolate(
    pred_mask_student[0], size=(224, 224), mode="bilinear", align_corners=False
)

mask_map_coach = pred_mask_coach_224[mask_frame_idx, 0]
mask_map_student = pred_mask_student_224[mask_frame_idx, 0]

save_mask_overlay(
    raw_coach[mask_frame_idx],
    mask_map_coach.detach().cpu().numpy().astype("float32"),
    os.path.join(save_dir, f"{coach_name}_mask_overlay.png"),
)
save_mask_overlay(
    raw_student[mask_frame_idx],
    mask_map_student.detach().cpu().numpy().astype("float32"),
    os.path.join(save_dir, f"{student_name}_mask_overlay.png"),
)

print("extracting plain DINO features for correspondence...")
feat_coach_hwc = extract_plain_dino_patch_features(
    plain_dino, px_coach, frame_idx=mask_frame_idx, patch_grid=PATCH_GRID
)
feat_student_hwc = extract_plain_dino_patch_features(
    plain_dino, px_student, frame_idx=mask_frame_idx, patch_grid=PATCH_GRID
)

print("building FMAP-based dense correspondence first...")
feat_coach_copca, feat_student_copca = co_pca_pair(
    feat_coach_hwc,
    feat_student_hwc,
    out_dim=COPCA_DIM,
)

fmap_model = FunctionalMap(
    src_data_hwc=feat_coach_copca,
    tgt_data_hwc=feat_student_copca,
    eigen_num=EIGEN_NUM,
)

src_to_tgt, src_score, src_margin, src_sim = fmap_model.get_pointwise_map()
tgt_to_src, tgt_score, tgt_sim = fmap_model.get_reverse_pointwise_map()

print("saving smooth full-image correspondence plot...")
save_smooth_fullimage_correspondence_plot(
    raw_source_frame=raw_coach[mask_frame_idx],
    raw_target_frame=raw_student[mask_frame_idx],
    src_to_tgt=src_to_tgt,
    out_path=os.path.join(save_dir, "smooth_fullimage_correspondence.png"),
    patch_grid=PATCH_GRID,
    upsample_to=IMG_SIZE,
)

# ------------------------------------------------------------
# restrict candidates using coach mask AFTER correspondence exists
# ------------------------------------------------------------
mask_patch_coach = F.interpolate(
    mask_map_coach.unsqueeze(0).unsqueeze(0),
    size=(PATCH_GRID, PATCH_GRID),
    mode="bilinear",
    align_corners=False,
).squeeze(0).squeeze(0)

mask_patch_student = F.interpolate(
    mask_map_student.unsqueeze(0).unsqueeze(0),
    size=(PATCH_GRID, PATCH_GRID),
    mode="bilinear",
    align_corners=False,
).squeeze(0).squeeze(0)

candidate_patch_idx = torch.nonzero(mask_patch_coach > mask_threshold, as_tuple=False)
candidate_patch_idx = candidate_patch_idx[:, 0] * PATCH_GRID + candidate_patch_idx[:, 1]

if candidate_patch_idx.numel() == 0:
    raise RuntimeError("no candidate source patches found from coach mask")

print(f"candidate masked source patches: {candidate_patch_idx.numel()}")

if mutual_only:
    mutual_keep = []
    for s_idx in candidate_patch_idx.tolist():
        t_idx = src_to_tgt[s_idx].item()
        s_back = tgt_to_src[t_idx].item()
        mutual_keep.append(s_back == s_idx)
    mutual_keep = torch.tensor(mutual_keep, device=device, dtype=torch.bool)
    candidate_patch_idx = candidate_patch_idx[mutual_keep]

if candidate_patch_idx.numel() == 0:
    raise RuntimeError("all candidates were removed by mutual consistency")

print(f"mutual-consistent candidates: {candidate_patch_idx.numel()}")

candidate_tgt_patch_idx = src_to_tgt[candidate_patch_idx]

coach_mask_flat = mask_patch_coach.reshape(-1)
student_mask_flat = mask_patch_student.reshape(-1)

target_mask_keep = student_mask_flat[candidate_tgt_patch_idx] > mask_threshold
candidate_patch_idx = candidate_patch_idx[target_mask_keep]
candidate_tgt_patch_idx = candidate_tgt_patch_idx[target_mask_keep]

if candidate_patch_idx.numel() == 0:
    raise RuntimeError("all candidates removed by student-side mask consistency")

print(f"student-mask-consistent candidates: {candidate_patch_idx.numel()}")

candidate_scores = (
    src_score[candidate_patch_idx]
    + 0.5 * src_margin[candidate_patch_idx]
    + 0.25 * coach_mask_flat[candidate_patch_idx]
    + 0.25 * student_mask_flat[candidate_tgt_patch_idx]
)

sorted_idx = torch.argsort(candidate_scores, descending=True)
candidate_patch_idx = candidate_patch_idx[sorted_idx]
candidate_tgt_patch_idx = candidate_tgt_patch_idx[sorted_idx]
candidate_scores = candidate_scores[sorted_idx]

if candidate_patch_idx.numel() > pre_fps_pool_size:
    candidate_patch_idx = candidate_patch_idx[:pre_fps_pool_size]
    candidate_tgt_patch_idx = candidate_tgt_patch_idx[:pre_fps_pool_size]
    candidate_scores = candidate_scores[:pre_fps_pool_size]

# ------------------------------------------------------------
# target-side dedup / local suppression
# ------------------------------------------------------------
kept_src = []
kept_tgt = []
kept_scores = []

occupied = torch.zeros((PATCH_GRID, PATCH_GRID), dtype=torch.bool, device=device)

for s_idx, t_idx, score in zip(candidate_patch_idx, candidate_tgt_patch_idx, candidate_scores):
    ty = (t_idx // PATCH_GRID).item()
    tx = (t_idx % PATCH_GRID).item()

    y0 = max(0, ty - target_radius)
    y1 = min(PATCH_GRID, ty + target_radius + 1)
    x0 = max(0, tx - target_radius)
    x1 = min(PATCH_GRID, tx + target_radius + 1)

    if occupied[y0:y1, x0:x1].any():
        continue

    kept_src.append(s_idx)
    kept_tgt.append(t_idx)
    kept_scores.append(score)
    occupied[y0:y1, x0:x1] = True

if len(kept_src) == 0:
    raise RuntimeError("no candidates left after target-side suppression")

candidate_patch_idx = torch.stack(kept_src)
candidate_tgt_patch_idx = torch.stack(kept_tgt)
candidate_scores = torch.stack(kept_scores)

# ------------------------------------------------------------
# FPS AFTER correspondence filtering
# ------------------------------------------------------------
src_patch_centers = patch_index_to_center_xy(
    candidate_patch_idx,
    patch_hw=(PATCH_GRID, PATCH_GRID),
    img_hw=(IMG_SIZE, IMG_SIZE),
    device=device,
)

if src_patch_centers.shape[0] > max_query_points:
    keep_pts = farthest_point_sampling_2d(src_patch_centers, max_query_points)

    selected_ids = []
    used = torch.zeros(src_patch_centers.shape[0], dtype=torch.bool, device=device)

    for pt in keep_pts:
        d = torch.sum((src_patch_centers - pt[None]) ** 2, dim=1)
        d[used] = float("inf")
        idx = torch.argmin(d)
        selected_ids.append(idx)
        used[idx] = True

    selected_ids = torch.stack(selected_ids)
    final_src_patch_idx = candidate_patch_idx[selected_ids]
    final_tgt_patch_idx = candidate_tgt_patch_idx[selected_ids]
else:
    final_src_patch_idx = candidate_patch_idx
    final_tgt_patch_idx = candidate_tgt_patch_idx

final_src_patch_idx = torch.unique(final_src_patch_idx)
final_tgt_patch_idx = src_to_tgt[final_src_patch_idx]

src_centers_224 = patch_index_to_center_xy(
    final_src_patch_idx,
    patch_hw=(PATCH_GRID, PATCH_GRID),
    img_hw=(IMG_SIZE, IMG_SIZE),
    device=device,
)

tgt_centers_224 = patch_index_to_center_xy(
    final_tgt_patch_idx,
    patch_hw=(PATCH_GRID, PATCH_GRID),
    img_hw=(IMG_SIZE, IMG_SIZE),
    device=device,
)

print(f"selected matched patch pairs after correspondence scoring: {src_centers_224.shape[0]}")

save_pair_overlay(
    raw_coach[mask_frame_idx],
    raw_student[mask_frame_idx],
    src_centers_224.detach().cpu(),
    tgt_centers_224.detach().cpu(),
    os.path.join(save_dir, "anchor_matched_points.png"),
)

# ------------------------------------------------------------
# free memory before cotracker
# ------------------------------------------------------------
del raft_model
del mask_model
del plain_dino
del flow_tensor
del out_coach
del out_student
torch.cuda.empty_cache()

print("loading cotracker for joint tracking...")
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

_, _, _, h_coach, w_coach = video_tensor_coach.shape
_, _, _, h_student, w_student = video_tensor_student.shape

y_coach = src_centers_224[:, 1]
x_coach = src_centers_224[:, 0]
y_student = tgt_centers_224[:, 1]
x_student = tgt_centers_224[:, 0]

y_coach_orig = y_coach.float() * ((h_coach - 1) / (IMG_SIZE - 1))
x_coach_orig = x_coach.float() * ((w_coach - 1) / (IMG_SIZE - 1))
t_coach = torch.full_like(x_coach_orig, float(query_frame_coach))
queries_coach = torch.stack([t_coach, x_coach_orig, y_coach_orig], dim=1).unsqueeze(0).to(device)

y_student_orig = y_student.float() * ((h_student - 1) / (IMG_SIZE - 1))
x_student_orig = x_student.float() * ((w_student - 1) / (IMG_SIZE - 1))
t_student = torch.full_like(x_student_orig, float(query_frame_student))
queries_student = torch.stack([t_student, x_student_orig, y_student_orig], dim=1).unsqueeze(0).to(device)

print(f"tracking {queries_coach.shape[1]} FMAP-linked points in both videos...")

pred_tracks_coach, pred_vis_coach = cotracker(video_tensor_coach, queries=queries_coach)
pred_tracks_student, pred_vis_student = cotracker(video_tensor_student, queries=queries_student)

save_data = {
    "coach": {
        "video_path": video_coach_path,
        "query_frame": int(query_frame_coach),
        "tracks": pred_tracks_coach.detach().cpu().numpy(),
        "visibility": pred_vis_coach.detach().cpu().numpy(),
        "fps": fps_coach,
    },
    "student": {
        "video_path": video_student_path,
        "query_frame": int(query_frame_student),
        "tracks": pred_tracks_student.detach().cpu().numpy(),
        "visibility": pred_vis_student.detach().cpu().numpy(),
        "fps": fps_student,
    },
    "anchor_matching": {
        "src_patch_idx": final_src_patch_idx.detach().cpu().numpy(),
        "tgt_patch_idx": final_tgt_patch_idx.detach().cpu().numpy(),
        "src_centers_224": src_centers_224.detach().cpu().numpy(),
        "tgt_centers_224": tgt_centers_224.detach().cpu().numpy(),
        "src_score": src_score[final_src_patch_idx].detach().cpu().numpy(),
        "src_margin": src_margin[final_src_patch_idx].detach().cpu().numpy(),
        "coach_mask_score": mask_patch_coach.reshape(-1)[final_src_patch_idx].detach().cpu().numpy(),
        "student_mask_score": mask_patch_student.reshape(-1)[final_tgt_patch_idx].detach().cpu().numpy(),
    },
}

tracks_path = os.path.join(save_dir, "paired_trajectories_fmap_sample_after.pkl")
with open(tracks_path, "wb") as f:
    pickle.dump(save_data, f)

print(f"saved paired trajectories to {tracks_path}")

vis_coach = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, fps=fps_coach)
vis_coach.visualize(video_tensor_coach, pred_tracks_coach, pred_vis_coach, filename=f"{coach_name}_tracks_fmap")

vis_student = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, fps=fps_student)
vis_student.visualize(video_tensor_student, pred_tracks_student, pred_vis_student, filename=f"{student_name}_tracks_fmap")

print("num queried coach points:", queries_coach.shape[1])
print("num queried student points:", queries_student.shape[1])
print("coach visible at query frame:", int((pred_vis_coach[0, query_frame_coach] > 0.5).sum().item()))
print("student visible at query frame:", int((pred_vis_student[0, query_frame_student] > 0.5).sum().item()))