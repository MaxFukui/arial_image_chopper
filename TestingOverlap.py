import rasterio
import geopandas as gpd

LABELS_DIR = "/home/corbusier/development/arial_image_chopper/Files/Labels"
main_path = LABELS_DIR + "/{file_name}.shp"
image_fp = "/home/corbusier/development/arial_image_chopper/Files/2018 - DJI P4 Pro/UPS_SUR_transparent_mosaic_group1.tif"
gdf_list = {
    "Background": main_path.format(file_name="Background"),
    "Grass": main_path.format(file_name="Grass"),
    "Bush": main_path.format(file_name="Bush"),
    "Vegetation": main_path.format(file_name="Vegetation"),
    "Foreground_UPS": main_path.format(file_name="Foreground_UPS"),
}

# Load the raster
with rasterio.open(image_fp) as src:
    print("Raster CRS:", src.crs)

# Load the shapefile
gdf = gpd.read_file(gdf_list['Grass'])
print("Shapefile CRS:", gdf.crs)

# Assuming 'src' is your rasterio opened raster
raster_crs = src.crs

# Reproject shapefile to match the raster CRS
gdf = gdf.to_crs(raster_crs)

