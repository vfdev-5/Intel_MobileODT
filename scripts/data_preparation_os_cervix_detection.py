import os
import sys
from datetime import datetime

# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)
          
from data_utils import RESOURCES_PATH, GENERATED_DATA, get_annotations
from image_utils import generate_label_images

sloth_annotations_filename = os.path.join(RESOURCES_PATH, 'cervix_os.json')
annotations = get_annotations(sloth_annotations_filename)
print("Found ", len(annotations), " annotations")

generate_label_images(annotations)
