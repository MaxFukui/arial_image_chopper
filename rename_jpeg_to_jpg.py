import os
import json


def rename_jpeg_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg"):
            base = os.path.splitext(filename)[0]
            new_name = base + ".jpg"
            os.rename(
                os.path.join(directory, filename), os.path.join(
                    directory, new_name)
            )
            print(f"Renamed {filename} to {new_name}")


if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to the directory containing your .jpeg files
    with open("config_ups_3_labels.json", "r") as config_file:
        config = json.load(config_file)
    directory_path = config['OUTPUT_DIR'] + "/tiles"
    rename_jpeg_to_jpg(directory_path)
