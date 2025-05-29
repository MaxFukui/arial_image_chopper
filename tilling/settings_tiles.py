import os
import imageio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.features import rasterize
from rasterio.windows import Window
from PIL import Image


def plot_sample(tile_path, mask_path):
    with rasterio.open(tile_path) as tile, rasterio.open(mask_path) as mask:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title('Tile')
        show(tile, ax=ax1, cmap='viridis')
        ax2.set_title('Mask')
        show(mask, ax=ax2, cmap='viridis')
        plt.show()


def tile_images(image_path, mask_path, tile_size, output_dir, prefix):
    with rasterio.open(image_path) as src, rasterio.open(mask_path) as src_mask:
        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                tile = src.read(window=window)
                mask_tile = src_mask.read(window=window)

                # Save the tiles
                tile_filename = os.path.join(
                    output_dir + "/tiff", f"{prefix}_tile_{i}_{j}.tif")
                mask_filename = os.path.join(
                    output_dir + "/tiff", f"{prefix}_mask_{i}_{j}.tif")

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


def save_color_mapped_mask(mask_path, color_mask_path, class_labels, colormap):
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)

    color_mask = np.zeros(
        (mask_data.shape[0], mask_data.shape[1], 3), dtype=np.uint8)

    for label, color in colormap.items():
        color_mask[mask_data == label] = color

    imageio.imwrite(color_mask_path, color_mask)
    print(f"Color-mapped mask saved at {color_mask_path}")


def create_mask(image_path, shapefiles, class_labels, output_mask_path):
    with rasterio.open(image_path) as src:
        out_meta = src.meta.copy()
        mask_data = np.zeros(src.shape, dtype=np.uint8)

        for label, path in shapefiles.items():
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs(src.crs)

            shapes = ((geom, class_labels[label]) for geom in gdf.geometry)
            rasterized = rasterize(
                shapes,
                out_shape=src.shape,
                fill=0,
                transform=src.transform,
                all_touched=True,
                dtype=np.uint8
            )
            mask_data = np.maximum(mask_data, rasterized)

        out_meta.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(output_mask_path, 'w', **out_meta) as out_raster:
            out_raster.write(mask_data, 1)

        print(f"Mask created with unique values: {np.unique(mask_data)}")
        plt.figure()
        plt.hist(mask_data.flatten(), bins=len(
            class_labels), range=(0, len(class_labels)))
        plt.title("Mask Value Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


def create_raster_mask(image_path, shapefiles, class_labels, output_mask_path):
    with rasterio.open(image_path) as src:
        out_meta = src.meta.copy()
        mask_data = np.zeros(src.shape, dtype=np.uint8)

        for label, path in shapefiles.items():
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs(src.crs)

            shapes = ((geom, class_labels[label]) for geom in gdf.geometry)
            rasterized = rasterize(
                shapes,
                out_shape=src.shape,
                fill=0,
                transform=src.transform,
                all_touched=True,
                dtype=np.uint8
            )
            mask_data = np.maximum(mask_data, rasterized)

        out_meta.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(output_mask_path, 'w', **out_meta) as out_raster:
            out_raster.write(mask_data, 1)

        print(f"Mask created with unique values: {np.unique(mask_data)}")
        plt.figure()
        plt.hist(mask_data.flatten(), bins=len(
            class_labels), range=(0, len(class_labels)))
        plt.title("Mask Value Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


def apply_color_map(mask_tile, colormap):
    height, width = mask_tile.shape
    color_mask_tile = np.zeros((height, width, 3), dtype=np.uint8)

    for label, color in colormap.items():
        color_mask_tile[mask_tile == label] = color

    return color_mask_tile


def tile_raster_images(image_path, mask_path, tile_size, output_dir, colormap, prefix):
    with rasterio.open(image_path) as src, rasterio.open(mask_path) as src_mask:
        for j in range(0, src.height, tile_size):
            for i in range(0, src.width, tile_size):
                window = Window(i, j, tile_size, tile_size)
                tile = src.read(window=window)
                mask_tile = src_mask.read(window=window)[0]

                color_mask_tile = apply_color_map(mask_tile, colormap)
                tile = np.moveaxis(tile, 0, -1)

                tile_filename = os.path.join(
                    output_dir + "/tiles", f"{prefix}_{i}_{j}.png")
                mask_filename = os.path.join(
                    output_dir + "/masks", f"{prefix}_{i}_{j}.png")

                print("Mask created", mask_filename)
                imageio.imwrite(tile_filename, tile)
                imageio.imwrite(mask_filename, color_mask_tile)


def transform_PNG_to_JPG(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # Open the .tif file
            png_path = os.path.join(directory, filename)
            with Image.open(png_path) as img:
                # Convert to .jpeg
                jpeg_filename = filename.replace('.png', '.jpg')
                jpeg_path = os.path.join(directory, jpeg_filename)
                img.convert('RGB').save(jpeg_path, 'JPEG')
                print(f'Converted: {filename} -> {jpeg_filename}')
    print('All files have been converted.')
