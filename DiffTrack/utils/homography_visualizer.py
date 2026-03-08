import cv2
import numpy as np
import os

def visualize_homography(img1_path, img2_path, bg_mask_path=None):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    img1 = cv2.resize(img1, (960, 520))
    img2 = cv2.resize(img2, (960, 520))
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    mask = None
    if bg_mask_path:
        pass

    pts1 = cv2.goodFeaturesToTrack(gray1, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    pts2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None)
    
    valid1 = pts1[status == 1]
    valid2 = pts2[status == 1]
    
    H, _ = cv2.findHomography(valid2, valid1, cv2.RANSAC, 3.0)
    h, w = img1.shape[:2]
    img2_warped = cv2.warpPerspective(img2, H, (w, h))
    
    diff_orig = cv2.absdiff(img1, img2)
    diff_warped = cv2.absdiff(img1, img2_warped)
    top_row = np.hstack((img2, img2_warped))
    bottom_row = np.hstack((diff_orig, diff_warped))
    vis_grid = np.vstack((top_row, bottom_row))
    
    cv2.putText(vis_grid, "Original Frame 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_grid, "Warped (Aligned) Frame 2", (970, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_grid, "Motion (Before Correction)", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(vis_grid, "Motion (After Correction)", (970, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out_dir = "homography_visualization"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "alignment_check.png")
    cv2.imwrite(out_path, vis_grid)
    print(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    # Point these to any two consecutive frames in your video folder
    frame0 = "videos/swim_3/frames_001.jpg" 
    frame1 = "videos/swim_3/frames_002.jpg"
    
    if os.path.exists(frame0) and os.path.exists(frame1):
        visualize_homography(frame0, frame1)
    else:
        print("Check your frame paths!")