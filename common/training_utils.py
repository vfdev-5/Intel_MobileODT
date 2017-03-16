
import numpy as np
import cv2

# Project
from data_utils import test_ids, type_to_index, type_1_ids, type_2_ids, type_3_ids
from image_utils import get_image_data


def get_trainval_id_type_lists(val_split=0.3, type_ids=(type_1_ids, type_2_ids, type_3_ids)):
    image_types = ["Type_1", "Type_2", "Type_3"]
    train_ll = [int(len(ids) * (1.0 - val_split)) for ids in type_ids]
    val_ll = [int(len(ids) * (val_split)) for ids in type_ids]

    count = 0
    train_id_type_list = []
    train_ids = [ids[:l] for ids, l in zip(type_ids, train_ll)]
    max_size = max(train_ll)
    while count < max_size:
        for l, ids, image_type in zip(train_ll, train_ids, image_types):
            image_id = ids[count % l]
            train_id_type_list.append((image_id, image_type))
        count += 1


    count = 0
    val_id_type_list = []
    val_ids = [ids[tl: tl +vl] for ids, tl, vl in zip(type_ids, train_ll, val_ll)]
    max_size = max(val_ll)
    while count < max_size:
        for l, ids, image_type in zip(val_ll, val_ids, image_types):
            image_id = ids[count % l]
            val_id_type_list.append((image_id, image_type))
        count += 1

    assert len(set(train_id_type_list) & set(val_id_type_list)) == 0, "WTF"

    print("Train dataset contains : ")
    print("-", train_ll, " images of corresponding types")
    print("Validation dataset contains : ")
    print("-", val_ll, " images of corresponding types")

    return train_id_type_list, val_id_type_list


def get_test_id_type_list():
    return [(image_id, 'Test') for image_id in test_ids]


def data_iterator(image_id_type_list, batch_size, image_size, verbose=0, test_mode=False):

    while True:
        X = np.zeros((batch_size, 3) + image_size, dtype=np.float32)
        Y = np.zeros((batch_size, 3), dtype=np.uint8)
        image_ids = np.empty((batch_size,), dtype=np.object)
        counter = 0
        for i, (image_id, image_type) in enumerate(image_id_type_list):

            img = get_image_data(image_id, image_type)
            img = cv2.resize(img, dsize=image_size[::-1])
            img = img.transpose([2 ,0 ,1])
            img = img.astype(np.float32) / 255.0

            X[counter, :, :, :] = img
            if test_mode:
                image_ids[counter] = image_id
            else:
                Y[counter, type_to_index[image_type]] = 1

            if verbose > 0:
                print("Image id/type:", image_id, image_type)

            counter += 1
            if counter == batch_size:
                yield (X, Y) if not test_mode else (X, Y, image_ids)
                X = np.zeros((batch_size, 3) + image_size, dtype=np.float32)
                Y = np.zeros((batch_size, 3), dtype=np.uint8)
                image_ids = np.empty((batch_size,), dtype=np.object)
                counter = 0

        if counter > 0:
            X = X[:counter ,: ,: ,:]
            Y = Y[:counter ,:]
            image_ids = image_ids[:counter]
            yield (X, Y) if not test_mode else (X, Y, image_ids)

        if test_mode:
            break