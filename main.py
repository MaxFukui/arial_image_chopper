import os
import json
from tilling.settings_tiles import (create_mask, save_color_mapped_mask,
                                    tile_images, plot_sample, tile_raster_images,
                                    transform_PNG_to_JPG
                                    )

# Load configuration
# with open('config_ups_3_labels.json', 'r') as config_file:
with open('config_ucdb_3_labels.json', 'r') as config_file:
    config = json.load(config_file)

LABELS_DIR = config['LABELS_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
PREFIX = config['prefix']
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR + "/masks", exist_ok=True)
os.makedirs(OUTPUT_DIR + "/tiles", exist_ok=True)
os.makedirs(OUTPUT_DIR + "/tiff", exist_ok=True)
image_fp = config['image_fp']
main_path = LABELS_DIR + "/{file_name}.shp"

shapefiles = {label: main_path.format(
    file_name=name) for label, name in config['shapefiles'].items()}
class_labels = config['class_labels']
colormap = {int(k): tuple(v) for k, v in config['colormap'].items()}

# Create the mask
output_mask_path = config['output_mask_path']
create_mask(image_fp, shapefiles, class_labels, output_mask_path)

# Save color-mapped mask
print("Saving color-mapped mask")
color_mask_path = config['color_mask_path']
save_color_mapped_mask(output_mask_path, color_mask_path,
                       class_labels, colormap)

# Tile the images
print("Tiling images")
tile_size = config['tile_size']
tile_images(image_fp, output_mask_path, tile_size,
            OUTPUT_DIR, PREFIX)

# Optionally check one tile and mask pair visually
# tile_sample_path = os.path.join(OUTPUT_DIR, "tile_0_0.tif")
# mask_sample_path = os.path.join(OUTPUT_DIR, "mask_0_0.tif")
# plot_sample(tile_sample_path, mask_sample_path)

tile_raster_images(image_fp, output_mask_path, tile_size,
                   OUTPUT_DIR, colormap, PREFIX)
transform_PNG_to_JPG(OUTPUT_DIR + "/tiles")
