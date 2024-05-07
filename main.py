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
# Define class labels with unique integers
class_labels = {
    "Background": 0,  # Typically, '0' is used for the background
    "Grass": 1,
    "Bush": 2,
    "Vegetation": 3,
    "Foreground_UPS": 4  # Assuming 'Foreground_UPS' is another class
}

def plot_mask(mask):
    # Create a color map
    cmap = plt.get_cmap('viridis', np.max(mask) - np.min(mask) + 1)
    plt.imshow(mask, cmap=cmap, vmin = np.min(mask)-0.5, vmax = np.max(mask)+0.5)
    # Add colorbar to show the label colors
    cbar = plt.colorbar(ticks=np.arange(np.min(mask), np.max(mask)+1))
    # cbar.ax.set_yticklabels(['Background', 'Grass', 'Bush', 'Vegetation', 'Foreground_UPS'])  # Update with your class names
    plt.grid(False)
    plt.show()


from shapely.geometry import box
import matplotlib.pyplot as plt

def create_composite_mask(src, shapes, class_labels, window, debug=True):
    mask_image = np.zeros((window.height, window.width), dtype=np.uint8)
    intermediate_masks = []

    for label_name, label_value in class_labels.items():
        if label_name in shapes:
            window_bounds = src.window_bounds(window)
            window_geom = box(*window_bounds)
            filtered_geometries = [shape for shape in shapes[label_name] if shape.intersects(window_geom)]
            if filtered_geometries:
                out_image, _ = mask(src, filtered_geometries, window, crop=False, filled=True, invert=False)
                if out_image.shape[1:3] == (window.height, window.width):
                    valid_mask = out_image[0] > 0
                    mask_image[valid_mask] = label_value  # Apply label value
                    if debug:
                        # Store a copy of the mask for this label for debugging
                        debug_mask = np.copy(mask_image)
                        intermediate_masks.append((label_name, debug_mask))

    if debug:
        # Plot each intermediate mask
        for name, dbg_mask in intermediate_masks:
            plt.figure(figsize=(5, 5))
            plt.title(f"Mask for {name}")
            plt.imshow(dbg_mask, cmap='viridis')
            plt.colorbar()
            plt.show()

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
                plot_image(tile)
                plot_mask(mask_tile)
                tiles.append(tile)
                mask_tiles.append(mask_tile)

    return tiles, mask_tiles


# Load shapefiles using Geopandas
## Create an array of file path geometry
main_path = LABELS_DIR + "/{file_name}.shp"
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
