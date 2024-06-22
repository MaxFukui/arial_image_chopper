import rasterio
from rasterio.mask import mask as riomask
import geopandas as gpd
import numpy as np
from rasterio.windows import from_bounds, Window
from shapely.geometry import box
import matplotlib.pyplot as plt
def plot_image(image):
    plt.imshow(image[0], cmap='viridis')
    plt.show()

def plot_out_image(out_image, label_name):
    plt.figure(figsize=(10, 10))
    plt.title(f"Mask Output for {label_name}")
    plt.imshow(out_image[0], cmap='viridis')  # Assuming out_image is a 3D array with band as the first dimension
    plt.colorbar()
    plt.show()

def plot_mask(mask):
Learning and Adapting Robust Features for Satellite Image Segmentation on Heterogeneous Data Sets
This paper addresses the problem of training a deep neural network for satellite image segmentation so that it can be deployed over images whose statistics differ from those used for training. For example, in postdisaster damage assessment, the tight time constraints make it impractical to train a network from scratch for each image to be segmented. We propose a convolutional encoder- decoder network able to learn visual representations of increasing semantic level as its depth increases, allowing it to generalize over a wider range of satellite images. Then, we propose two additional methods to improve the network performance over each specific image to be segmented. First, we observe that updating the batch normalization layers' statistics over the target image improves the network performance without human intervention. Second, we show that refining a trained network over a few samples of the image boosts the network performance with minimal human intervention. We evaluate our architecture over three data sets of satellite images, showing the state-of-the-art performance in binary segmentation of previously unseen images and competitive performance with respect to more complex techniques in a multiclass segmentation task.
    unique_values = np.unique(mask)
    print("Plotting mask with unique values:", unique_values)  # Debugging statement

    n_colors = max(len(unique_values), 1)  # Ensure at least one color
    cmap = plt.get_cmap('viridis', n_colors)

    fig, ax = plt.subplots()
    cax = ax.imshow(mask, cmap=cmap, vmin=np.min(unique_values)-0.5, vmax=np.max(unique_values)+0.5)
    cbar = plt.colorbar(cax, ticks=np.arange(np.min(unique_values), np.max(unique_values)+1))

    # Dynamically set the colorbar labels based on the unique values
    cbar_labels = [label_dict.get(value, 'Unknown') for value in np.arange(np.min(unique_values), np.max(unique_values) + 1)]
    cbar.ax.set_yticklabels(cbar_labels)

    plt.grid(False)
    plt.show()

def create_mask_for_each_class(src, shapes, class_labels, window):
    """Generate separate mask for each class and merge them into a composite mask."""
    mask_image = np.zeros((window.height, window.width), dtype=np.uint8)
    window_transform = src.window_transform(window)  # Get the transform for the window

    for label_name, geometries in shapes.items():
        filtered_geometries = [geom for geom in geometries if geom.intersects(box(*src.window_bounds(window)))]
        if filtered_geometries:  # Check if there are any filtered geometries
            out_image, out_transform = riomask(src, filtered_geometries, crop=True, all_touched=True)

            # Check dimensions of out_image against the window dimensions
            if out_image.shape[1] != window.height or out_image.shape[2] != window.width:
                print(f"Dimension mismatch for {label_name}: expected ({window.height}, {window.width}), got {out_image.shape[1:]}. Adjusting dimensions.")
                # Adjust the dimensions of out_image to match the window, if necessary
                adjusted_image = np.zeros((1, window.height, window.width), dtype=out_image.dtype)
                min_height = min(window.height, out_image.shape[1])
                min_width = min(window.width, out_image.shape[2])
                adjusted_image[0, :min_height, :min_width] = out_image[0, :min_height, :min_width]
                out_image = adjusted_image

            if out_image.any():  # Check if there's any non-zero data in the output image
                valid_mask = out_image[0] > 0
                mask_image[valid_mask] = class_labels[label_name]
                print(f"Applied {label_name} mask with label {class_labels[label_name]}")
        else:
            print(f"No valid intersections for {label_name}")

    return mask_image

def generate_dataset(image_path, gdf_list, class_labels, tile_size):
    tiles = []
    mask_tiles = []

    with rasterio.open(image_path) as src:
        shapes = {label: gpd.read_file(gdf_list[label]).geometry for label in gdf_list}
        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                tile = src.read(window=window)
                mask_tile = create_mask_for_each_class(src, shapes,class_labels, window)
                plot_mask(mask_tile)
                plot_image(tile)
                tiles.append(tile)
                mask_tiles.append(mask_tile)
                print(f"Processed tile at {i}, {j}")

    return tiles, mask_tiles

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

gdf_list = {
    "Background": main_path.format(file_name="Background"),
    "Grass": main_path.format(file_name="Grass"),
    "Bush": main_path.format(file_name="Bush"),
    "Vegetation": main_path.format(file_name="Vegetation"),
    "Foreground_UPS": main_path.format(file_name="Foreground_UPS"),
}

tiles, mask_tiles = generate_dataset(image_fp, gdf_list, class_labels, tile_size=1024)
