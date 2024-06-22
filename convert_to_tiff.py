import os
from PIL import Image

# Directory containing the tif files
directory = '/home/corbusier/development/arial_image_chopper/output/tiff_files/Tiles'

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.tif'):
        # Open the .tif file
        tif_path = os.path.join(directory, filename)
        with Image.open(tif_path) as img:
            # Convert to .jpeg
            jpeg_filename = filename.replace('.tif', '.jpeg')
            jpeg_path = os.path.join(directory, jpeg_filename)
            img.convert('RGB').save(jpeg_path, 'JPEG')
            print(f'Converted: {filename} -> {jpeg_filename}')

print('All files have been converted.')
