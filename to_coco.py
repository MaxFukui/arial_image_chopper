import os
import json
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

def create_coco_json(image_dir, mask_dir, json_file_path):
    coco_output = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{
            'id': 1,
            'name': 'example_category',
            'supercategory': 'none',
        }]
    }

    image_id = 1
    annotation_id = 1

    for filename in os.listdir(image_dir):
        if filename.endswith('.tif'):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)  # Assuming mask has same filename

            # Open the image to get dimensions
            image = Image.open(image_path)
            width, height = image.size

            # Create image info
            image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_path), (width, height))
            coco_output['images'].append(image_info)

            # Create mask annotation
            mask = np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale
            binary_mask = np.where(mask > 0, 1, 0)  # Assuming non-zero pixels are objects
            print("Shape of binary_mask:", binary_mask.shape)
            print("Data type of elements:", type(binary_mask[0][0]))

            annotation_info = pycococreatortools.create_annotation_info(
                annotation_id, image_id, {'id': 1, 'is_crowd': False},
                binary_mask, (width, height), tolerance=2
            )

            if annotation_info is not None:
                coco_output['annotations'].append(annotation_info)
                annotation_id += 1

            image_id += 1

    with open(json_file_path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)



# Specify directories and output path
image_directory = '/home/corbusier/development/arial_image_chopper/teste_coco/image'
mask_directory = '/home/corbusier/development/arial_image_chopper/teste_coco/mask'
output_coco_json = '/home/corbusier/development/arial_image_chopper/teste_coco'

create_coco_json(image_directory, mask_directory, output_coco_json)
