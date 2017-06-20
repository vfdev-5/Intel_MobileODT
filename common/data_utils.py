
import os
import json 
import platform
from glob import glob
import numpy as np

project_common_path = os.path.dirname(__file__)

DATA_PATH = os.path.abspath(os.path.join(project_common_path, "..", "input"))
INPUT_PATH = DATA_PATH


TRAIN_DATA = os.path.join(DATA_PATH, "train")
TEST_DATA = os.path.join(DATA_PATH, "test")
ADDITIONAL_DATA = os.path.join(DATA_PATH, "additional")

RESOURCES_PATH = os.path.abspath(os.path.join(project_common_path, "..", "resources"))
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


if len(type_1_ids) > 0 and len(type_2_ids) > 0 and len(type_3_ids) > 0:
    # ## TYPE 1
    # Remove bad images:
    type_1_ids.remove('1339')  # corrupted image
    ids_to_remove = [
        # Corrupted
        '3068',
        # Size = 0
        '5893',
        # No cervix
        '4706', '2030', '4065', '4702', '6360', '746',
        '6180', '6181', '6182', '3997', '5983', '5985', '5948',
        '3659', '3058', '3504', '4042', '4043', '6361',
        # Blurry
        '4874', '4701', '4041', '4687', '4684', '2835', '2150',
        '4557', '6748', '4875', '3968', '3973', '3974', '3975', '5769',
    ]
    for i in ids_to_remove:
        try:
            additional_type_1_ids.remove(i)
        except ValueError as e:
            print("Type_1", e, i)

    # ## TYPE 2
    # Remove bad images:
    ids_to_remove = [
        # Size = 0
        '2845', '5892', '7',
        # No cervix
        '1813', '1879', '2310', '3002', '3542', '1720', '592', '3086', '2172', '1338',
        '1982', '1988', '1744', '2145', '2936', '6599', '6600', '6601',
        # Blurry
        '1839', '6829', '2717', '2311', '2329', '3450', '3386', '908', '3103', '6294', '6295',
        '6853', '6854', '6855', '6872', '6874', '6896', '879', '1230',
    ]
    for i in ids_to_remove:
        try:
            additional_type_2_ids.remove(i)
        except ValueError as e:
            print("Type_2", e, i)

    # ## TYPE 3
    # Remove bad images:
    ids_to_remove = [
        # No cervix
        '5691', '5684', '5792', '5683', '5714', '5688', '5690', '5954', '6342', '3507', '5677',
        '5685', '327', '4331', '3081', '1588', '5401',
        # Blurry
        '1618', '2036', '2095', '2255', '2828', '5696', '5897', '1767', '4763', '4764', '4936',
        '5605', '5610', '5702', '5755', '1063', '2637', '6885', '4550', '4611', '5330', '5433',
        '5437', '5426', '4224', '5651', '5689', '5705', '5967', '6347', '4173', '4354', '4189',
        '4575', '4777', '2490',
    ]
    for i in ids_to_remove:
        try:
            additional_type_3_ids.remove(i)
        except ValueError as e:
            print("Type_3", e, i)


# Test data
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = [s[len(TEST_DATA)+1:-4] for s in test_files]

assert len(test_ids) > 0, "No test data found in folder : '%s'" % TEST_DATA

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


def get_id_type_list_from_annotations(annotations, select=None):
    id_type_list = []
    select_set = set(select) if select is not None else set()

    for annotation in annotations:

        if select is not None:
            class_types = [str(a['class']) for a in annotation['annotations']]
            if len(set(class_types) & select_set) == 0:
                continue

        image_name = annotation['filename']
        image_id = os.path.basename(image_name)[:-4]
        splt = os.path.split(os.path.dirname(image_name))
        if os.path.basename(splt[0]).lower() == "train":
            image_type = splt[1]
        elif os.path.basename(splt[0]).lower() == "additional":
            image_type = "A" + splt[1]
        else:
            raise Exception("Unknown type : %s" % os.path.basename(splt[0]))
        id_type_list.append((image_id, image_type))
    return id_type_list


def compute_type_distribution(id_type_list):
    types = (('Type_1', 'AType_1'), ('Type_2', 'AType_2'), ('Type_3', 'AType_3'))
    ll = len(id_type_list)
    out = [0.0, 0.0, 0.0]
    for i, ts in enumerate(types):
        for t in ts:
            out[i] += (id_type_list[:, 1] == t).sum()
        out[i] *= 1.0 / ll
    return out


def to_set(id_type_array):
    return set([(i[0], i[1]) for i in id_type_array.tolist()])


def remove_green_imagery(id_type_list):
    output = list(id_type_list)
    green_id_type_path = os.path.join(GENERATED_DATA, "green_id_type_list.npz")
    assert os.path.exists(green_id_type_path)
    green_id_type_list = np.load(green_id_type_path)['green_id_type_list'].astype(str)
    green_id_type_set = to_set(green_id_type_list)
    common_green_id_type_list = list(set(id_type_list) & green_id_type_set)
    for k in common_green_id_type_list:
        output.remove(k)
    return output
