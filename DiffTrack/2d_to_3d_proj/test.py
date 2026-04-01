import numpy as np

def generate_3d_bridge(data_path, output_name, frame_idx=0, dino_mask=None):
    # load npz data
    data = np.load(data_path)
    
    # extract arrays
    img = data['images'][frame_idx]          # [h, w, 3]
    depth = data['depths'][frame_idx].astype(np.float32) # [h, w]
    intrinsics = data['intrinsic']           # [3, 3]
    c2w = data['cam_c2w'][frame_idx]         # [4, 4]
    
    h, w = depth.shape

    # create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()
    
    # apply dino mask if provided
    if dino_mask is not None:
        mask = dino_mask.flatten() > 0.5
        u, v, z = u[mask], v[mask], z[mask]
        colors = img.reshape(-1, 3)[mask]
    else:
        colors = img.reshape(-1, 3)

    # back-project to camera space
    # x = (u - cx) * z / fx ; y = (v - cy) * z / fy
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x_c = (u - cx) * z / fx
    y_c = (v - cy) * z / fy
    z_c = z
    
    pts_cam = np.stack([x_c, y_c, z_c, np.ones_like(z)], axis=1) # [n, 4]

    # transform to world space using c2w
    pts_world = (pts_cam @ c2w.T)[:, :3] # [n, 3]

    # write ply file for visualization
    with open(output_name, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts_world)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        
        for p, c in zip(pts_world, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    print(f"saved {len(pts_world)} points to {output_name}")

data_dir = "2d_to_3d_proj/mega-sam/outputs_cvd/benchpress_sgd_cvd_hr.npz"
generate_3d_bridge(data_dir, "2d_to_3d_proj/output_frame_0.ply", frame_idx=0)