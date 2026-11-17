import os
import glob
import rasterio
import numpy as np
from PIL import Image

def slice_all_available_tiffs(output_dir, patch_size=256, stride=128):
    """
    Dynamically auto-detects all .tif files in the current folder,
    extracts their internal layers, corrects the dynamic range, 
    and slices them into standard ML patches.
    """
    # Create the output patch bucket if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Automatically scan the folder for any file ending in .tif
    tiff_files = glob.glob("*.tif")
    
    if not tiff_files:
        print("[ERROR] Zero .tif files detected in this folder!")
        print("Fix: Move your city .tif files into the exact same folder as this script.")
        return

    print(f"[INFO] Pipeline initiated. Found {len(tiff_files)} target file(s): {tiff_files}")
    total_patches_generated = 0

    # Loop through every file discovered
    for file_path in tiff_files:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0] # Extracts city name without '.tif'
        
        print(f"\n--- Processing File: {file_name} ---")
        
        try:
            with rasterio.open(file_path) as src:
                print(f"[-] Resolution: {src.width}x{src.height} pixels | Internal Layers: {src.count}")
                
                if src.count < 3:
                    print(f"[WARNING] Skipping {file_name}: Needs at least 3 hidden bands, found {src.count}")
                    continue
                
                # Extract the 3 core spectral layers from the file sandwich
                layer_1 = src.read(1) 
                layer_2 = src.read(2)
                layer_3 = src.read(3)
                
            print("[-] Compiling layers into standard 3D array layout...")
            stacked = np.stack((layer_3, layer_2, layer_1), axis=-1)
            
            print("[-] Applying Min-Max contrast stretch to eliminate the black picture issue...")
            max_pixel_value = np.max(stacked)
            if max_pixel_value == 0: 
                max_pixel_value = 1023 # Standard sensor maximum fallback
                
            # Scale the low raw numbers safely into the viewable 0-255 range
            normalized_img = (stacked / max_pixel_value * 255).astype(np.uint8)
            h, w, _ = normalized_img.shape
            
            file_patch_count = 0
            print(f"[-] Slicing map grid into {patch_size}x{patch_size} blocks...")
            
            # Sliding window execution loop
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = normalized_img[y:y+patch_size, x:x+patch_size, :]
                    
                    # Ignore border patches that contain mostly empty black space
                    if np.mean(patch) < 10:
                        continue
                        
                    # Save patch with a distinct prefix matching the specific city file
                    patch_img = Image.fromarray(patch)
                    patch_img.save(os.path.join(output_dir, f"{base_name}_patch_{file_patch_count}.png"))
                    file_patch_count += 1
                    total_patches_generated += 1
                    
            print(f"[SUCCESS] Finished {file_name}. Generated {file_patch_count} clear image patches.")
            
        except Exception as e:
            print(f"[ERROR] Failed to read {file_name}. Reason: {str(e)}")
            
    print(f"\n==================================================")
    print(f"[MASTER SUCCESS] Data refinery run complete.")
    print(f"Total patches waiting in bucket: {total_patches_generated}")
    print(f"Destination: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # Target directory structure matching your setup
    OUTPUT_FOLDER = "./data/patches"
    slice_all_available_tiffs(OUTPUT_FOLDER)