import os
import glob
import numpy as np
from PIL import Image

def purge_black_edges(patch_dir, black_pixel_tolerance=5, max_black_ratio=0.15):
    """
    Deletes patches that contain too much pure black 'NoData' space from the edge of the map.
    """
    all_patches = glob.glob(os.path.join(patch_dir, "*.png"))
    print(f"[-] Scanning {len(all_patches)} patches for NoData edge cuts...")
    
    deleted_count = 0
    
    for patch_path in all_patches:
        try:
            # Convert to Grayscale to easily measure pure darkness
            img = Image.open(patch_path).convert('L') 
            img_array = np.array(img)
            
            # Count how many pixels are essentially pitch black (values 0 to 5)
            total_pixels = img_array.shape[0] * img_array.shape[1]
            black_pixels = np.sum(img_array <= black_pixel_tolerance)
            
            # If more than 15% of the patch is the black void, nuke it
            if (black_pixels / total_pixels) > max_black_ratio:
                os.remove(patch_path)
                deleted_count += 1
                
        except Exception as e:
            print(f"Error processing {patch_path}: {e}")
            
    print(f"[✓] Edge Purge Complete.")
    print(f"[-] Deleted {deleted_count} patches containing black edges.")
    print(f"[-] Perfect, full-data patches remaining: {len(all_patches) - deleted_count}")

if __name__ == "__main__":
    PATCH_FOLDER = "./data/lissiv_patches"
    purge_black_edges(PATCH_FOLDER)