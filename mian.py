import rasterio
import imageio
from rasterio.mask import mask
from rasterio.windows import from_bounds, Window
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image):
    plt.imshow(image[0], cmap='viridis')
    plt.show()
# Define class labels with unique integers
class_labels = {
    "Background": 0,  # Typically, '0' is used for the background
    "Grass": 1,
    "Bush": 2,
    "Vegetation": 3,
    "Foreground_UPS": 4  # Assuming 'Foreground_UPS' is another class
}

def create_composite_mask(src, shapes, class_labels, window):
    # Initialize a zero array for the mask with the same dimensions as the window
    mask_image = np.zeros((window.height, window.width), dtype=np.uint8)

    # Apply each shapefile mask
    for label_name, label_value in class_labels.items():
        if label_name in shapes:
            window_bounds = src.window_bounds(window)
            window_geom = box(*window_bounds)  # Convert bounds to a geometry
            filtered_geometries = [shape for shape in shapes[label_name] if shape.intersects(window_geom)]
            if filtered_geometries:  # Check if there are any geometries to mask
                out_image, _ = mask(src, filtered_geometries, window, crop=False, filled=True, invert=False)
                # Reshape out_image if necessary
                if out_image.shape[1:3] != (window.height, window.width):
                    continue  # or handle differently, like padding or cropping
                # Only apply the mask where the out_image mask is true
                valid_mask = out_image[0] > 0
                if valid_mask.shape == mask_image.shape:
                    mask_image[valid_mask] = label_value
                else:
                    continue  # Handle cases where mask shapes do not match

    return mask_image

def generate_dataset(image_path, gdf_list, class_labels, tile_size):
    tiles = []
    mask_tiles = []

    with rasterio.open(image_path) as src:
        # Create masks for each class
        shapes = {label: [] for label in class_labels.keys()}
        for label_name in gdf_list:
            gdf = gpd.read_file(gdf_list[label_name])
            shapes[label_name] = [geom for geom in gdf.geometry]

        # Process the image by tiles
        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                tile = src.read(window=window)
                mask_tile = create_composite_mask(src, shapes, class_labels, window)

                tiles.append(tile)
                mask_tiles.append(mask_tile)

    return tiles, mask_tiles


# Load shapefiles using Geopandas
## Create an array of file path geometry
main_path = "C:/Users/max/Documents/{file_name}.shp"
# gdf_list = ["Background", "Grass", "Bush", "Vegetation", "Foreground_UPS"]
# gdf_list = ["Background", "Grass", "Bush", "Vegetation", "Foreground_UPS"]
gdf_list = {
    "Background": main_path.format(file_name="Background"),
    "Grass": main_path.format(file_name="Grass"),
    "Bush": main_path.format(file_name="Bush"),
    "Vegetation": main_path.format(file_name="Vegetation"),
    "Foreground_UPS": main_path.format(file_name="Foreground_UPS"),
}

# Path to your orthophoto
image_fp = "D:/development/UPS/2018 - DJI P4 Pro/UPS_SUR_transparent_mosaic_group1.tif"

tiles, mask_tiles = generate_dataset(image_fp, gdf_list, class_labels, tile_size=256)

def save_tiles(tiles, mask_tiles, output_dir):
    for i, (tile, mask_tile) in enumerate(zip(tiles, mask_tiles)):
        # Assuming tile and mask_tile are numpy arrays and tile has shape (bands, height, width)
        # Convert from (bands, height, width) to (height, width, bands) if necessary and save only the first band for simplicity
        image = np.moveaxis(tile, 0, -1)[:, :, 0]  # Adjust this if your tiles have more than one band
        mask_image = mask_tile  # Already should be in (height, width) format

        # Save the image and mask
        imageio.imwrite(f'{output_dir}/tile_{i}.png', image)
        imageio.imwrite(f'{output_dir}/mask_{i}.png', mask_image)


# Example usage
output_dir = "D:/development/image_choper/output"
save_tiles(tiles, mask_tiles, output_dir)
