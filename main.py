import rasterio
import imageio
from rasterio.mask import mask
from rasterio.windows import from_bounds, Window
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

LABELS_DIR = "/home/corbusier/development/arial_image_chopper/Files/Labels"
OUTPUT_DIR = "/home/corbusier/development/arial_image_chopper/output"

def plot_image(image):
    plt.imshow(image[0], cmap='viridis')
    plt.show()

def plot_out_image(out_image, label_name):
    plt.figure(figsize=(10, 10))
    plt.title(f"Mask Output for {label_name}")
    plt.imshow(out_image[0], cmap='viridis')  # Assuming out_image is a 3D array with band as the first dimension
    plt.colorbar()
    plt.show()

class_labels = {
    "Background": 0,  # Typically, '0' is used for the background
    "Grass": 1,
    "Bush": 2,
    "Vegetation": 3,
    "Foreground_UPS": 4  # Assuming 'Foreground_UPS' is another class
}


import matplotlib.pyplot as plt

def plot_mask(mask):
    label_dict = {
        0: 'Background',
        1: 'Grass',
        2: 'Bush',
        3: 'Vegetation',
        4: 'Foreground_UPS'
    }
    unique_values = np.unique(mask)
    print("Plotting mask with unique values:", unique_values)  # Debugging statement

    n_colors = max(len(unique_values), 1)  # Ensure at least one color
    cmap = plt.get_cmap('viridis', n_colors)

    fig, ax = plt.subplots()
    cax = ax.imshow(mask, cmap=cmap, vmin=np.min(unique_values)-0.5, vmax=np.max(unique_values)+0.5)
    cbar = plt.colorbar(cax, ticks=np.arange(np.min(unique_values), np.max(unique_values)+1))

    cbar_labels = [label_dict.get(value, '') for value in unique_values]
    cbar.ax.set_yticklabels(cbar_labels)

    plt.grid(False)
    plt.show()

# Example usage, assuming 'mask_image' is your mask array
# plot_mask(mask_image)

from shapely.geometry import box
import matplotlib.pyplot as plt

import rasterio
from rasterio.mask import mask as riomask
from shapely.geometry import box

def create_composite_mask(src, shapes, class_labels, window):
    mask_image = np.zeros((window.height, window.width), dtype=np.uint8)
    window_transform = src.window_transform(window)

    for label_name, label_value in class_labels.items():
        if label_name in shapes:
            window_bounds = src.window_bounds(window)
            window_geom = box(*window_bounds)
            filtered_geometries = [shape for shape in shapes[label_name] if shape.intersects(window_geom)]

            if filtered_geometries:
                out_image, out_transform = mask(src, filtered_geometries, crop=True)
                # Print out summary statistics of out_image
                print(f"out_image stats for {label_name}: min={out_image.min()}, max={out_image.max()}, mean={out_image.mean()}")
                plot_out_image(out_image, label_name)
                print(f"out_image shape: {out_image.shape}")

                if out_image.shape[1] != window.height or out_image.shape[2] != window.width:
                    continue  # Ensuring dimension match

                valid_mask = out_image[0] > 0
                if np.any(valid_mask):
                    print(f"Valid mask data present for {label_name}")
                else:
                    print(f"No valid mask data to update for {label_name}")

                mask_image[valid_mask] = label_value
            else:
                print(f"No geometries intersect the window for {label_name}")

    return mask_image

def generate_dataset(image_path, gdf_list, class_labels, tile_size):
    tiles = []
    mask_tiles = []

    with rasterio.open(image_path) as src:
        shapes = {label: [] for label in class_labels.keys()}
        for label_name in gdf_list:
            gdf = gpd.read_file(gdf_list[label_name])
            shapes[label_name] = [geom for geom in gdf.geometry]

        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                tile = src.read(window=window)
                mask_tile = create_composite_mask(src, shapes, class_labels, window)
                print(f"Unique mask values for tile at ({i},{j}): {np.unique(mask_tile)}")  # Debugging line
                plot_image(tile)
                plot_mask(mask_tile)
                imageio.imwrite(f'{OUTPUT_DIR}/mask_{i}.png', mask_tile)
                tiles.append(tile)
                mask_tiles.append(mask_tile)

    return tiles, mask_tiles


# load shapefiles using geopandas
## create an array of file path geometry
main_path = LABELS_DIR + "/{file_name}.shp"

gdf_list = {
    "Background": main_path.format(file_name="Background"),
    "Grass": main_path.format(file_name="Grass"),
    "Bush": main_path.format(file_name="Bush"),
    "Vegetation": main_path.format(file_name="Vegetation"),
    "Foreground_UPS": main_path.format(file_name="Foreground_UPS"),
}

# path to your orthophoto
image_fp = "/home/corbusier/development/arial_image_chopper/Files/2018 - DJI P4 Pro/UPS_SUR_transparent_mosaic_group1.tif"

tiles, mask_tiles = generate_dataset(image_fp, gdf_list, class_labels, tile_size=1024)

def save_tiles(tiles, mask_tiles, output_dir):
    # tiles_mask_dict = zip(tiles, mask_tiles)
    # for i, (tile, mask_tile) in enumerate():
    for i in range(0, 6):
        tile = tiles[i]
        mask_tile = mask_tiles[i]
        print(tiles[i])
        print(mask_tiles[i])

        # Assuming tile and mask_tile are numpy arrays and tile has shape (bands, height, width)
        # Convert from (bands, height, width) to (height, width, bands) if necessary and save only the first band for simplicity
        image = np.moveaxis(tile, 0, -1)[:, :, 0]  # Adjust this if your tiles have more than one band
        mask_image = mask_tile  # Already should be in (height, width) format

        # Save the image and mask
        imageio.imwrite(f'{output_dir}/tile_{i}.png', image)
        imageio.imwrite(f'{output_dir}/mask_{i}.png', mask_image)
# Example usage
save_tiles(tiles, mask_tiles, OUTPUT_DIR)
