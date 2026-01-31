import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_coco_subset(target_dir, num_images=2000):
    """Download a subset of COCO 2017 val images (smaller than train)"""
    os.makedirs(target_dir, exist_ok=True)
    
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(target_dir, "coco_val.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading COCO val2017 subset (approx 800MB)...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    
    print("Extracting images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract only the first num_images
        file_list = [f for f in zip_ref.namelist() if f.endswith('.jpg')]
        for file in tqdm(file_list[:num_images], desc="Extracting"):
            zip_ref.extract(file, target_dir)
            
    # Move files to the target_dir root for easier access
    extract_subdir = os.path.join(target_dir, "val2017")
    for filename in os.listdir(extract_subdir):
        shutil.move(os.path.join(extract_subdir, filename), os.path.join(target_dir, filename))
    
    os.rmdir(extract_subdir)
    # os.remove(zip_path) # Keep zip for now in case of restart
    print(f"Successfully downloaded {num_images} content images to {target_dir}")

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_images")
    download_coco_subset(target)
