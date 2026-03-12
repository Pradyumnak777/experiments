import os
import cv2


def mp4_to_frames(video_name):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)

	if os.path.isabs(video_name):
		video_path = video_name
		rel_video_no_ext = os.path.splitext(os.path.basename(video_name))[0]
	else:
		video_path = os.path.join(project_root, video_name)
		rel_video_no_ext = os.path.splitext(video_name)[0]

	output_dir = os.path.join(project_root, "videos", rel_video_no_ext)
	os.makedirs(output_dir, exist_ok=True)

	if not os.path.exists(video_path):
		raise FileNotFoundError(f"Video file not found: {video_path}")

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	frame_idx = 0
	print(f"Extracting frames from {video_path} to {output_dir}...")

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_path = os.path.join(output_dir, f"frames_{frame_idx:03d}.jpg")
		cv2.imwrite(frame_path, frame)
		frame_idx += 1

	cap.release()
	print(f"Saved {frame_idx} frames to {output_dir}")


if __name__ == "__main__":
	video_name = "UCF_Rep/val/v_Kayaking_g23_c06.mp4"  
	mp4_to_frames(video_name)
