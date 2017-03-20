
import numpy as np
import cv2

from keras.preprocessing.image import random_rotation, random_shift, flip_axis

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


def compute_mean_std_images(image_id_type_list, output_size=(224, 224), feature_wise=False, verbose=0):
    """
    Method to compute mean/std input image
    :return: mean_image, std_image
    """
    nc = 3
    ll = len(image_id_type_list)
    # Init mean/std images
    mean_image = np.zeros(tuple(output_size[::-1]) + (nc,), dtype=np.float32)
    std_image = np.zeros(tuple(output_size[::-1]) + (nc,), dtype=np.float32)
    for i, (image_id, image_type) in enumerate(image_id_type_list):
        if verbose > 0:
            print("Image id/type:", image_id, image_type, "| ", i+1, "/", ll)

        img = get_image_data(image_id, image_type)
        if img.dtype.kind is not 'u':
            if verbose > 0:
                print("Image is corrupted. Id/Type:", image_id, image_type)
            continue
        img = cv2.resize(img, dsize=output_size[::-1])
        if feature_wise:
            mean_image += np.mean(img, axis=(0, 1))
            std_image += np.std(img, axis=(0, 1))
        else:
            mean_image += img
            std_image += np.power(img, 2.0)

    mean_image *= 1.0 / ll
    std_image *= 1.0 / ll
    if not feature_wise:
        std_image -= np.power(mean_image, 2.0)
        std_image = np.sqrt(std_image)
    return mean_image, std_image


def data_augmentation(X, Y,
                      hflip=True, vflip=True,
                      random_transformations=True):
    yield X, Y
    if hflip:
        _X = X.copy()
        for i in range(_X.shape[0]):
            _X[i, :, :, :] = flip_axis(X[i, :, :, :], axis=-1)
        yield (_X, Y)

    if vflip:
        _X = X.copy()
        for i in range(_X.shape[0]):
            _X[i, :, :, :] = flip_axis(X[i, :, :, :], axis=-2)
        yield (_X, Y)

    if hflip and vflip:
        _X = X.copy()
        for i in range(_X.shape[0]):
            _X[i, :, :, :] = flip_axis(flip_axis(X[i, :, :, :], axis=-2), axis=-1)
        yield (_X, Y)

    if random_transformations:
        _X = X.copy()
        for i in range(_X.shape[0]):
            _X[i, :, :, :] = random_rotation(X[i, :, :, :], rg=180)
        yield (_X, Y)

        _X = X.copy()
        for i in range(_X.shape[0]):
            _X[i, :, :, :] = random_shift(X[i, :, :, :], wrg=0.2, hrg=0.2)
        yield (_X, Y)


def data_iterator(image_id_type_list, batch_size, image_size, verbose=0, test_mode=False, data_augmentation_fn=None):

    assert len(image_id_type_list) > 0, "Input data image/type list is empty"

    while True:
        X = np.zeros((batch_size, 3) + image_size, dtype=np.float32)
        Y = np.zeros((batch_size, 3), dtype=np.uint8)
        image_ids = np.empty((batch_size,), dtype=np.object)
        counter = 0
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", counter)

            img = get_image_data(image_id, image_type)
            if img.dtype.kind is not 'u':
                if verbose > 0:
                    print("Image is corrupted. Id/Type:", image_id, image_type)
                continue
            img = cv2.resize(img, dsize=image_size[::-1])
            img = img.transpose([2, 0, 1])
            img = img.astype(np.float32) / 255.0

            X[counter, :, :, :] = img
            if test_mode:
                image_ids[counter] = image_id
            else:
                Y[counter, type_to_index[image_type]] = 1

            counter += 1
            if counter == batch_size:
                if data_augmentation_fn is not None and not test_mode:
                    for _X, _Y in data_augmentation_fn(X, Y):
                        yield (_X, _Y)
                else:
                    yield (X, Y) if not test_mode else (X, Y, image_ids)

                X = np.zeros((batch_size, 3) + image_size, dtype=np.float32)
                Y = np.zeros((batch_size, 3), dtype=np.uint8)
                image_ids = np.empty((batch_size,), dtype=np.object)
                counter = 0

        if counter > 0:
            X = X[:counter, :, :, :]
            Y = Y[:counter, :]
            image_ids = image_ids[:counter]
            if data_augmentation_fn is not None and not test_mode:
                for _X, _Y in data_augmentation_fn(X, Y):
                    yield (_X, _Y)
            else:
                yield (X, Y) if not test_mode else (X, Y, image_ids)

        if test_mode:
            break
