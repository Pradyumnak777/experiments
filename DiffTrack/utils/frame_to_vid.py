import os
import glob
import cv2


def frames_to_mp4(video_name, fps=30):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    frames_dir = os.path.join(project_root, "videos", video_name)
    output_dir = os.path.join(project_root, "vids_mp4")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_name}.mp4")

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Writing {len(frame_paths)} frames to {output_path}...")
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    video_name = "swim_3"  # change this
    frames_to_mp4(video_name, fps=20)
