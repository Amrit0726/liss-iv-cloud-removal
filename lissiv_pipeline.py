import os
import glob
import numpy as np
import rasterio
from PIL import Image

def process_lissiv_scene(scene_folder, output_dir, patch_size=256, stride=128):
    """
    Processes one LISS-IV scene folder containing BAND2.tif, BAND3.tif, BAND4.tif
    and slices it into patches.
    BAND2 = Green, BAND3 = Red, BAND4 = NIR
    We use RGB = (BAND3, BAND2, BAND4) for a natural-looking composite.
    """
    band2_path=glob.glob(os.path.join(scene_folder, "*BAND2.tif"))
    band3_path=glob.glob(os.path.join(scene_folder, "*BAND3.tif"))
    band4_path=glob.glob(os.path.join(scene_folder, "*BAND4.tif"))

    # Also try simple BAND2.tif naming
    if not band2_path:
        band2_path=glob.glob(os.path.join(scene_folder, "BAND2.tif"))
        band3_path=glob.glob(os.path.join(scene_folder, "BAND3.tif"))
        band4_path=glob.glob(os.path.join(scene_folder, "BAND4.tif"))

    if not band2_path or not band3_path or not band4_path:
        print(f"[WARNING] Missing band files in {scene_folder}. Skipping.")
        return 0

    print(f"\n--- Processing: {os.path.basename(scene_folder)} ---")

    try:
        with rasterio.open(band2_path[0]) as b2:
            green=b2.read(1).astype(np.float32)
        with rasterio.open(band3_path[0]) as b3:
            red=b3.read(1).astype(np.float32)
        with rasterio.open(band4_path[0]) as b4:
            nir=b4.read(1).astype(np.float32)

        print(f"[-] Band shape: {red.shape}")

        # Stack as RGB = Red, Green, NIR (false color — better for vegetation/land)
        stacked=np.stack([red, green, nir], axis=-1)

        # Per-channel normalization for better contrast
        normalized=np.zeros_like(stacked, dtype=np.uint8)
        for i in range(3):
            channel=stacked[:,:,i]
            p2=np.percentile(channel, 2)
            p98=np.percentile(channel, 98)
            if p98>p2:
                channel=np.clip((channel-p2)/(p98-p2)*255, 0, 255)
            else:
                channel=np.zeros_like(channel)
            normalized[:,:,i]=channel.astype(np.uint8)

        h, w, _=normalized.shape
        scene_name=os.path.basename(scene_folder)
        patch_count=0

        print(f"[-] Slicing {h}x{w} image into {patch_size}x{patch_size} patches...")

        for y in range(0, h-patch_size+1, stride):
            for x in range(0, w-patch_size+1, stride):
                patch=normalized[y:y+patch_size, x:x+patch_size, :]

                # Skip mostly black or mostly white patches
                mean_val=np.mean(patch)
                if mean_val < 10 or mean_val > 245:
                    continue

                patch_img=Image.fromarray(patch)
                patch_img.save(os.path.join(output_dir, f"{scene_name}_patch_{patch_count}.png"))
                patch_count+=1

        print(f"[✓] Generated {patch_count} patches from {scene_name}")
        return patch_count

    except Exception as e:
        print(f"[ERROR] Failed to process {scene_folder}: {e}")
        return 0

def process_all_lissiv_scenes(root_dir, output_dir, patch_size=256, stride=128):
    """
    Scans root_dir for all LISS-IV scene folders and processes each one.
    Works with both:
    - root_dir/scene_folder/BAND2.tif  (single scene)
    - root_dir/scene_folder/subfolder/BAND2.tif  (nested)
    """
    os.makedirs(output_dir, exist_ok=True)
    total_patches=0

    # Find all folders containing BAND2.tif or *BAND2.tif
    scene_folders=set()
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if 'BAND2' in f.upper() and f.endswith('.tif'):
                scene_folders.add(root)

    if not scene_folders:
        print(f"[ERROR] No LISS-IV band files found in {root_dir}")
        print("Make sure your unzipped scene folders are inside this directory.")
        return

    print(f"[INFO] Found {len(scene_folders)} LISS-IV scene(s) to process.")

    for folder in sorted(scene_folders):
        count=process_lissiv_scene(folder, output_dir, patch_size, stride)
        total_patches+=count

    print(f"\n{'='*50}")
    print(f"[DONE] Total patches generated: {total_patches}")
    print(f"[DONE] Saved to: {os.path.abspath(output_dir)}")

if __name__=="__main__":
    # Put all your unzipped scene folders inside ./lissiv_data/
    # Structure should be:
    # lissiv_data/
    #   R2F08SEP2024.../
    #       BAND2.tif
    #       BAND3.tif
    #       BAND4.tif
    #   R2F05JUN2023.../
    #       BAND2.tif
    #       ...

    
    ROOT_DIR="./data/lissiv_data"
    OUTPUT_DIR="./data/test_patches"
    process_all_lissiv_scenes(ROOT_DIR, OUTPUT_DIR)
