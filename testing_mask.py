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