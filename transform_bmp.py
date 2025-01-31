from PIL import Image
import os
import numpy as np

# Folder containing the .png files
input_folder = "/home/docker/Work/Test_manuscrit/9"
output_folder = "/home/docker/Work/Test_manuscrit_v3"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        # Full path to the input file
        input_path = os.path.join(input_folder, filename)
        
        # Full path to the output file
        output_path = os.path.join(output_folder, "9_" + os.path.splitext(filename)[0][-1] + ".bmp")
        
        # Open the image
        with Image.open(input_path) as img:
            # Convert image to RGB mode
            img_rgb = img.convert("RGB")
            
            # Convert image to numpy array for normalization
            img_array = np.array(img_rgb)
            
            # Ensure pixel values are within the 0-255 range (clip them if necessary)
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # Convert back to an image after processing the array
            img_rgb_normalized = Image.fromarray(img_array)
            
            # Save as BMP
            img_rgb_normalized.save(output_path, "BMP")

print("All .png files have been converted to .bmp!")

