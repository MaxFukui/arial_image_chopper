import os

# Directory containing the tif files
directory = "/home/corbusier/development/arial_image_chopper/output/png_files/masks"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.startswith("mask_") and filename.endswith(".png"):
        # Create the new filename by removing 'mask_' prefix
        new_filename = filename.replace("mask_", "", 1)
        # Get the full path for both the old and new filenames
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {filename} -> {new_filename}")

print("All files have been renamed.")
