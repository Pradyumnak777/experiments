import pickle, numpy as np, matplotlib.pyplot as plt, os

def plot_with_ids_save(pkl_path, out_path=None):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    traj = data["trajectory"][0].astype(np.float32)  # (T, N, 2)
    T, N, _ = traj.shape

    if "video_shape" in data:  #this is for canvas size, cuz there may be more points than 
        _, H, W = data["video_shape"]
    else:
        W = int(np.nanmax(traj[...,0])) + 1
        H = int(np.nanmax(traj[...,1])) + 1

    traj = np.nan_to_num(traj, nan=-1.0, posinf=-1.0, neginf=-1.0)

    t = np.linspace(0, 1, T, dtype=np.float32) #time color in the plot
    xs = traj[..., 0].reshape(-1)
    ys = traj[..., 1].reshape(-1)
    cs = np.repeat(t[:, None], N, axis=1).reshape(-1)

    p0 = traj[0]

    plt.figure(figsize=(10, 7))
    plt.scatter(xs, ys, c=cs, s=14, cmap="viridis")
    plt.scatter(p0[:, 0], p0[:, 1], c="red", s=55) #these are the points in the starting frame...

    for i in range(N):
        plt.text(p0[i, 0] + 6, p0[i, 1] + 6, str(i), fontsize=9, color="red") #write frame/id next to start point

    plt.xlim(0, W - 1)
    plt.ylim(H - 1, 0)
    plt.colorbar(label="time")
    plt.title(f"all point trajectories with ids (N={N}, T={T})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if out_path is None:
        out_path = pkl_path.replace(".pkl", "_traj_with_ids.png")

    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


if __name__ == "__main__":
    pkl_file = "DiffTrack/custom_vid_output/layer[17]_timestep[49]_noiseFalse/trajectories/001_trajectory.pkl"
    plot_with_ids_save(pkl_file)
