import json
import tilling.settings_tiles as transform_PNG_to_JPG

with open('config_ucdb_3_labels.json', 'r') as config_file:
    config = json.load(config_file)
transform_PNG_to_JPG.transform_PNG_to_JPG(config['OUTPUT_DIR'] + "/tiles")
