import os
import zipfile
import subprocess
import shutil
from huggingface_hub import hf_hub_download
from tqdm import tqdm

hf_repo = "quchenyuan/UCF101-ZIP"
zip_filename = "UCF-101.zip"
annotation_dir = "annotations_ucfrep"
target_dir = "UCF_Rep"

def get_video_split_map():
    video_to_split = {}
    for subset in ["train", "val"]: 
        folder_path = os.path.join(annotation_dir, subset)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith('.mat'):
                    video_base = f.replace('.mat', '')
                    video_to_split[f"{video_base}.avi"] = subset
    return video_to_split

if __name__ == "__main__":
    #get zip from hf cache
    zip_path = hf_hub_download(repo_id=hf_repo, filename=zip_filename, repo_type="dataset")
    split_map = get_video_split_map()
    
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(target_dir, subset), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        files_to_process = [f for f in all_files if os.path.basename(f) in split_map]

        for zip_file_path in tqdm(files_to_process, desc="Converting to MP4 (No Audio)"):
            vid_filename_avi = os.path.basename(zip_file_path)
            vid_filename_mp4 = vid_filename_avi.replace('.avi', '.mp4')
            target_subset = split_map[vid_filename_avi]
            
            #extract raw avi temporarily
            temp_avi_path = zip_ref.extract(zip_file_path, target_dir)
            final_mp4_path = os.path.join(target_dir, target_subset, vid_filename_mp4)

            #convert to mp4 and drop audio with -an
            cmd = [
                'ffmpeg', '-y', '-i', temp_avi_path, 
                '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast', 
                '-an', final_mp4_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            #delete the temp avi file
            os.remove(temp_avi_path)

    #cleanup the empty ucf101 folder structure from zip
    raw_folder = os.path.join(target_dir, "UCF101")
    if os.path.exists(raw_folder):
        shutil.rmtree(raw_folder)

    print(f"done! files in {target_dir}/train and {target_dir}/val")