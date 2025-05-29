import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from rasterio.windows import Window
import os
import matplotlib.pyplot as plt
import imageio


# Function to create the mask
def create_mask(image_path, shapefiles, class_labels, output_mask_path):
    with rasterio.open(image_path) as src:
        out_meta = src.meta.copy()
        # ensure dtype matches expected label range
        mask_data = np.zeros(src.shape, dtype=np.uint8)

        for label, path in shapefiles.items():
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs(src.crs)  # Align CRS with the raster

            shapes = ((geom, class_labels[label]) for geom in gdf.geometry)
            rasterized = rasterize(
                shapes,
                out_shape=src.shape,
                fill=0,
                transform=src.transform,
                all_touched=True,
                dtype=np.uint8
            )
            mask_data = np.maximum(mask_data, rasterized)  # Combine masks

        # Save the mask using the same metadata as the source, but with 1 band
        out_meta.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(output_mask_path, 'w', **out_meta) as out_raster:
            out_raster.write(mask_data, 1)

        print(f"Mask created with unique values: {np.unique(mask_data)}")
        # Plot histogram
        plt.figure()
        plt.hist(mask_data.flatten(), bins=len(
            class_labels), range=(0, len(class_labels)))
        plt.title("Mask Value Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


# Function to save a color-mapped version of the mask
def save_color_mapped_mask(mask_path, color_mask_path, class_labels):
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)

    # Create a colormap for visualization
    colormap = {
        0: (0, 0, 0),  # Black for Background
        1: (255, 0, 0),  # Red for Grass
        2: (0, 255, 0),  # Green for Bush
        3: (0, 0, 255),  # Blue for Vegetation
        4: (0, 0, 0)  # Yellow for Foreground_UPS
    }

    color_mask = np.zeros(
        (mask_data.shape[0], mask_data.shape[1], 3), dtype=np.uint8)

    for label, color in colormap.items():
        color_mask[mask_data == label] = color

    # Save the color-mapped mask
    imageio.imwrite(color_mask_path, color_mask)
    print(f"Color-mapped mask saved at {color_mask_path}")


# Function to apply color map to a mask tile
def apply_color_map(mask_tile, colormap):
    height, width = mask_tile.shape
    color_mask_tile = np.zeros((height, width, 3), dtype=np.uint8)

    for label, color in colormap.items():
        color_mask_tile[mask_tile == label] = color

    return color_mask_tile


# Function to tile the images
def tile_images(image_path, mask_path, tile_size, output_dir):
    colormap = {
        0: (0, 0, 0),  # Black for Background
        1: (255, 0, 0),  # Red for Grass
        2: (0, 255, 0),  # Green for Bush
        3: (0, 0, 255),  # Blue for Vegetation
        4: (255, 255, 0)  # Yellow for Foreground_UPS
    }

    with rasterio.open(image_path) as src, rasterio.open(mask_path) as src_mask:
        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                tile = src.read(window=window)
                mask_tile = src_mask.read(window=window)[
                    0]  # Read first band for mask

                # Apply color map to the mask tile
                color_mask_tile = apply_color_map(mask_tile, colormap)

                # Convert tile to (height, width, bands) for saving as PNG
                tile = np.moveaxis(tile, 0, -1)

                # Save the tiles as PNG
                tile_filename = os.path.join(
                    output_dir + "/tiles", f"{i}_{j}.png")
                mask_filename = os.path.join(
                    output_dir + "/masks", f"{i}_{j}.png")

                print("Mask created: " + mask_filename)
                imageio.imwrite(tile_filename, tile)
                imageio.imwrite(mask_filename, color_mask_tile)


# Define paths and shapefiles
LABELS_DIR = "/home/corbusier/development/arial_image_chopper/Files/Labels"
OUTPUT_DIR = "/home/corbusier/development/arial_image_chopper/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR+"/masks", exist_ok=True)
os.makedirs(OUTPUT_DIR+"/tiles", exist_ok=True)
image_fp = "/home/corbusier/development/arial_image_chopper/Files/2018 - DJI P4 Pro/UPS_SUR_transparent_mosaic_group1.tif"
main_path = LABELS_DIR + "/{file_name}.shp"

shapefiles = {
    "Background": main_path.format(file_name="Background"),
    "Grass": main_path.format(file_name="Grass"),
    "Bush": main_path.format(file_name="Bush"),
    "Vegetation": main_path.format(file_name="Vegetation"),
    "Foreground_UPS": main_path.format(file_name="Foreground_UPS"),
}

class_labels = {
    "Background": 0,
    "Grass": 1,
    "Bush": 2,
    "Vegetation": 3,
    "Foreground_UPS": 4
}

# Create the mask
output_mask_path = "/home/corbusier/development/arial_image_chopper/mask.tif"
create_mask(image_fp, shapefiles, class_labels, output_mask_path)

# Save color-mapped mask
color_mask_path = "/home/corbusier/development/arial_image_chopper/color_mask.png"
save_color_mapped_mask(output_mask_path, color_mask_path, class_labels)

# Tile the images
tile_size = 512
tile_images(image_fp, output_mask_path, tile_size, OUTPUT_DIR)

# Check one tile and mask pair visually
tile_sample_path = os.path.join(OUTPUT_DIR + '/masks', "0_0.png")
mask_sample_path = os.path.join(OUTPUT_DIR + '/tiles', "0_0.png")


def plot_sample(tile_path, mask_path):
    tile = imageio.imread(tile_path)
    mask = imageio.imread(mask_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title('Tile')
    ax1.imshow(tile)
    ax2.set_title('Mask')
    ax2.imshow(mask)
    plt.show()


plot_sample(tile_sample_path, mask_sample_path)
# trigger a command to make a sound when the script finishes
