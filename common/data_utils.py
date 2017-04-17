
import os
import json 
import platform
from glob import glob


if 'c001' in platform.node():
    DATA_PATH = "/data/kaggle"
    INPUT_PATH = os.path.abspath("../input")
else:
    DATA_PATH = os.path.abspath("../input")
    INPUT_PATH = DATA_PATH


TRAIN_DATA = os.path.join(DATA_PATH, "train")
TEST_DATA = os.path.join(DATA_PATH, "test")
ADDITIONAL_DATA = os.path.join(DATA_PATH, "additional")

RESOURCES_PATH = os.path.abspath("../resources")
GENERATED_DATA = os.path.join(INPUT_PATH, 'generated')

if not os.path.exists(GENERATED_DATA):
    os.makedirs(GENERATED_DATA)

type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = [s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files]
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = [s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files]
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = [s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files]

additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "Type_1", "*.jpg"))
additional_type_1_ids = [s[len(os.path.join(ADDITIONAL_DATA, "Type_1"))+1:-4] for s in additional_type_1_files]
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "Type_2", "*.jpg"))
additional_type_2_ids = [s[len(os.path.join(ADDITIONAL_DATA, "Type_2"))+1:-4] for s in additional_type_2_files]
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "Type_3", "*.jpg"))
additional_type_3_ids = [s[len(os.path.join(ADDITIONAL_DATA, "Type_3"))+1:-4] for s in additional_type_3_files]


# Remove bad images:
type_1_ids.remove('1339')  # corrupted image
ids_to_remove = [
    # Corrupted
    '3068',
    # Size = 0
    '5893',
    # No cervix
    '4706',     '2030',    '4065',    '4702',    '6360',    '746',
    # Blurry
    '4874',     '4701',    '4041',    '4687',    '4684',    '2835',    '2150',
]
try:
    for i in ids_to_remove:
        additional_type_1_ids.remove(i)
except ValueError as e:
    print(e, i)


ids_to_remove = [
    # No cervix
    '5691', '5684', '5792', '5683', '5714', '5688', '5690', '5954', '6342', '3507', '5677',
    '5685', '327',
    # Blurry
    '1618', '2036', '2095', '2255', '2828', '5696', '5897', '1767', '4763', '4764', '4936',
    '5605', '5610', '5702', '5755', '1063', '2637', '6885', '4550', '4611', '5330', '5433',
    '5437', '5426', '4224', '5651', '5689', '5705', '5967', '6347'
]
try:
    for i in ids_to_remove:    
        additional_type_3_ids.remove(i)
except ValueError as e:
    print(e, i)


# Test data
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = [s[len(TEST_DATA)+1:-4] for s in test_files]

type_to_index = {
    "Type_1": 0,
    "Type_2": 1,
    "Type_3": 2,
    "AType_1": 0,
    "AType_2": 1,
    "AType_3": 2,
}


def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type
    """
    ext = 'jpg'
    check_dir = False
    if image_type == "Type_1" or \
        image_type == "Type_2" or \
        image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or \
          image_type == "AType_2" or \
          image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type[1:])
    elif image_type == 'trainval_label_0':
        # Manual os/servix segmentation
        data_path = os.path.join(GENERATED_DATA, 'trainval_labels_0')
        ext = 'npz'
        check_dir = True
    elif 'os_cervix_label_' in image_type:
        # Results of U-net predictions of os/cervix segmentation, with a seed
        data_path = os.path.join(GENERATED_DATA, image_type)
        ext = 'npz'
        check_dir = True
    elif image_type == 'os_cervix_bbox':
        # Results of ensembling of U-net predictions of os/cervix segmentation as bbox
        data_path = os.path.join(GENERATED_DATA, 'os_cervix_bbox')
        ext = 'npz'     
        check_dir = True
    elif image_type == 'Type_1_os' or \
          image_type == 'Type_2_os' or \
          image_type == 'Type_3_os':
        # Original Image masked with 'os_cervix_label':os and cropped/resized to 224x224
        t = image_type[:6]
        data_path = os.path.join(GENERATED_DATA, 'train', t, 'os')
        check_dir = True
    elif image_type == 'Test_os':
        # Original Image masked with 'os_cervix_label':os and cropped/resized to 224x224
        t = image_type[:6]
        data_path = os.path.join(GENERATED_DATA, 'test', 'os')
        check_dir = True
    elif image_type == 'Type_1_cervix' or \
          image_type == 'Type_2_cervix' or \
          image_type == 'Type_3_cervix':
        # Original Image masked with 'os_cervix_label':cervix and cropped/resized to 224x224
        t = image_type[:6]
        data_path = os.path.join(GENERATED_DATA, 'train', t, 'cervix')
        check_dir = True
    elif image_type == 'Test_cervix':
        # Original Image masked with 'os_cervix_label':cervix and cropped/resized to 224x224
        t = image_type[:6]
        data_path = os.path.join(GENERATED_DATA, 'test', 'cervix')
        check_dir = True
    elif image_type == 'trainval_label_gray':
        data_path = os.path.join(GENERATED_DATA, 'trainval_labels_gray')
        ext = 'png'
        check_dir = True
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def get_annotations(filename):
    """
    Annotation file created by SLOTH application
    
    :return: ndarray of dicts 
        {
            "annotations": [
                {
                    "class": "os",
                    "height": 10.0,
                    "type": "rect",
                    "width": 20.0,
                    "x": 52.0,
                    "y": 48.0
                },
                {
                    "class": "cervix",
                    "height": 275.0,
                    "type": "rect",
                    "width": 300.0,
                    "x": 10.0,
                    "y": 5.0
                }],
            "class": "image",
            "filename": "train/Type_1/590.jpg"
        }
    """
    labels = []
    with open(filename, 'r') as reader:
        str_data = ''.join(reader.readlines())
        raw_data = json.loads(str_data)
        for item in raw_data:
            if len(item['annotations']) > 0:
                labels.append(item)
    return labels
