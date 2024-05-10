import rasterio
from rasterio.plot import show
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from rasterio.windows import Window
import imageio


def create_mask(image_path, shapefiles, class_labels):
    # Load the base image
    with rasterio.open(image_path) as src:
        # Create a template for the mask
        mask_data = np.zeros(src.shape, dtype=np.uint8)

        # Rasterize each shapefile layer onto the mask
        for label, path in shapefiles.items():
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs(src.crs)  # Ensure CRS matches
            shapes = ((geom, class_labels[label]) for geom in gdf.geometry)
            mask_data += rasterize(shapes, out_shape=src.shape, fill=0, transform=src.transform, all_touched=True,
                                   dtype=np.uint8)

        # Save the mask
        out_meta = src.meta.copy()
        out_meta.update(dtype=rasterio.uint8, count=1)

        with rasterio.open('mask.tiff', 'w', **out_meta) as out_raster:
            out_raster.write(mask_data, 1)


def tile_images(image_path, mask_path, tile_size, output_dir):
    with rasterio.open(image_path) as src, rasterio.open(mask_path) as src_mask:
        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                tile = src.read(window=window)
                mask_tile = src_mask.read(window=window)

                # Save the tiles
                out_path = f"{output_dir}/tile_{i}_{j}.tif"
                out_mask_path = f"{output_dir}/mask_{i}_{j}.tif"

                with rasterio.open(out_path, 'w', **src.meta) as out_tile:
                    out_tile.write(tile)

                with rasterio.open(out_mask_path, 'w', **src_mask.meta) as out_mask_tile:
                    out_mask_tile.write(mask_tile)


# Example usage
# Define class labels
class_labels = {
    "Background": 0,
    "Grass": 1,
    "Bush": 2,
    "Vegetation": 3,
    "Foreground_UPS": 4
}

# Define paths and shapefiles
LABELS_DIR = "/home/corbusier/development/arial_image_chopper/Files/Labels"
OUTPUT_DIR = "/home/corbusier/development/arial_image_chopper/output"
image_fp = "/home/corbusier/development/arial_image_chopper/Files/2018 - DJI P4 Pro/UPS_SUR_transparent_mosaic_group1.tif"
main_path = LABELS_DIR + "/{file_name}.shp"

shapefiles = {
    "Background": main_path.format(file_name="Background"),
    "Grass": main_path.format(file_name="Grass"),
    "Bush": main_path.format(file_name="Bush"),
    "Vegetation": main_path.format(file_name="Vegetation"),
    "Foreground_UPS": main_path.format(file_name="Foreground_UPS"),
}

# create_mask(image_fp, shapefiles, class_labels)
# tile_images(image_fp, 'path_to_mask.tif', 1024, OUTPUT_DIR)

import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

def create_mask(image_path, shapefiles, class_labels):
    with rasterio.open(image_path) as src:
        out_meta = src.meta.copy()
        mask_data = np.zeros(src.shape, dtype=np.uint8)  # ensure dtype matches expected label range

        for label, path in shapefiles.items():
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs(src.crs)  # Align CRS with the raster

            shapes = ((geom, class_labels[label]) for geom in gdf.geometry)
            rasterized = rasterize(shapes, out_shape=src.shape, fill=0, transform=src.transform, all_touched=True, dtype=np.uint8)
            mask_data = np.maximum(mask_data, rasterized)  # use maximum to avoid overwriting with 0

        # Save the mask using the same metadata as the source, but with 1 band
        out_meta.update(dtype=rasterio.uint8, count=1)
        with rasterio.open('mask.tif', 'w', **out_meta) as out_raster:
            out_raster.write(mask_data, 1)

# Example usage
create_mask(image_fp, shapefiles, class_labels)

# After creating, check the mask:
mask_check = rasterio.open('mask.tif').read(1)
print(np.unique(mask_check))

import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from rasterio.windows import Window
import os
import matplotlib.pyplot as plt

# Function to create the mask
def create_mask(image_path, shapefiles, class_labels, output_mask_path):
    with rasterio.open(image_path) as src:
        out_meta = src.meta.copy()
        mask_data = np.zeros(src.shape, dtype=np.uint8)  # ensure dtype matches expected label range

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
        plt.hist(mask_data.flatten(), bins=len(class_labels), range=(0, len(class_labels)))
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
        0: (0, 0, 0),        # Black for Background
        1: (255, 0, 0),      # Red for Grass
        2: (0, 255, 0),      # Green for Bush
        3: (0, 0, 255),      # Blue for Vegetation
        4: (255, 255, 0)     # Yellow for Foreground_UPS
    }

    color_mask = np.zeros((mask_data.shape[0], mask_data.shape[1], 3), dtype=np.uint8)

    for label, color in colormap.items():
        color_mask[mask_data == label] = color

    # Save the color-mapped mask
    imageio.imwrite(color_mask_path, color_mask)
    print(f"Color-mapped mask saved at {color_mask_path}")

# Function to tile the images
def tile_images(image_path, mask_path, tile_size, output_dir):
    with rasterio.open(image_path) as src, rasterio.open(mask_path) as src_mask:
        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                tile = src.read(window=window)
                mask_tile = src_mask.read(window=window)

                # Save the tiles
                tile_filename = os.path.join(output_dir, f"tile_{i}_{j}.tif")
                mask_filename = os.path.join(output_dir, f"mask_{i}_{j}.tif")

                tile_meta = src.meta.copy()
                tile_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": rasterio.windows.transform(window, src.transform)
                })

                mask_meta = src_mask.meta.copy()
                mask_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": rasterio.windows.transform(window, src_mask.transform)
                })

                with rasterio.open(tile_filename, 'w', **tile_meta) as out_tile:
                    out_tile.write(tile)

                with rasterio.open(mask_filename, 'w', **mask_meta) as out_mask_tile:
                    out_mask_tile.write(mask_tile)

# Define paths and shapefiles
LABELS_DIR = "/home/corbusier/development/arial_image_chopper/Files/Labels"
OUTPUT_DIR = "/home/corbusier/development/arial_image_chopper/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
tile_size = 1024
tile_images(image_fp, output_mask_path, tile_size, OUTPUT_DIR)

# Check one tile and mask pair visually
tile_sample_path = os.path.join(OUTPUT_DIR, "tile_0_0.tif")
mask_sample_path = os.path.join(OUTPUT_DIR, "mask_0_0.tif")

def plot_sample(tile_path, mask_path):
    with rasterio.open(tile_path) as tile, rasterio.open(mask_path) as mask:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title('Tile')
        show(tile, ax=ax1, cmap='viridis')
        ax2.set_title('Mask')
        show(mask, ax=ax2, cmap='viridis')
        plt.show()

plot_sample(tile_sample_path, mask_sample_path)
