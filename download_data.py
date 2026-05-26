import gdown
import os
from PIL import Image
import pillow_heif

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

folder_id = "10Nwtmb-LYcPT63RFIe61IIgV7ebjEv7l"

try:
    # print("Downloading files from Google Drive...")
    # gdown.download_folder(
    #     f"https://drive.google.com/drive/folders/{folder_id}",
    #     output="photos/",
    #     quiet=False
    # )
    
    # Convert HEIC files to JPG
    print("\nConverting HEIC files to JPG...")
    heic_files = [f for f in os.listdir("Old GP Photos/") if f.lower().endswith((".heic", ".heif"))]
    
    for heic_file in heic_files:
        heic_path = os.path.join("Old GP Photos/", heic_file)
        jpg_path = os.path.join("Old GP Photos/", os.path.splitext(heic_file)[0] + ".jpg")
        
        try:
            image = Image.open(heic_path)
            image = image.convert("RGB")
            image.save(jpg_path, "JPEG", quality=95)
            print(f"✓ Converted: {heic_file} → {os.path.basename(jpg_path)}")
            os.remove(heic_path)  # Remove original HEIC file
        except Exception as e:
            print(f"✗ Error converting {heic_file}: {e}")
    
    print("\nDone! All HEIC files have been converted to JPG.")
    
except Exception as e:
    print(f"Download failed: {e}")
    print("Wait a few minutes and try again")