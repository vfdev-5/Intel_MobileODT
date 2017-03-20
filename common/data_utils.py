
import os
import platform
from glob import glob


if 'c001' in platform.node():
    DATA_PATH = "/data/kaggle"
else:
    DATA_PATH = "../input"

TRAIN_DATA = os.path.join(DATA_PATH, "train")
TEST_DATA = os.path.join(DATA_PATH, "test")
ADDITIONAL_DATA = os.path.join(DATA_PATH, "additional")

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

test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = [s[len(TEST_DATA)+1:-4] for s in test_files]

type_to_index = {
    "Type_1": 0,
    "Type_2": 1,
    "Type_3": 2,
}


def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type
    """
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
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))
