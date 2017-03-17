
#
# Test all images
#
import os
import datetime

# Project
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common'))
from data_utils import type_1_ids, type_2_ids, type_3_ids, test_ids
from image_utils import get_image_data
from training_utils import get_trainval_id_type_lists, get_test_id_type_list, data_iterator


print("\n=========================")
print("Training dataset: ")
print("- type 1: ", len(type_1_ids))
print("- type 2: ", len(type_2_ids))
print("- type 3: ", len(type_3_ids))

print("Test dataset: ")
print("- ", len(test_ids))
print("=========================\n")


def train(train_id_type_list, val_id_type_list, batch_size=16, nb_epochs=10, image_size=(224, 224)):
    samples_per_epoch = 2048
    nb_val_samples = 512

    if not os.path.exists('weights'):
        os.mkdir('weights')

    print("Training parameters: ", batch_size, nb_epochs, samples_per_epoch, nb_val_samples)

    train_iter = data_iterator(train_id_type_list, batch_size=batch_size, image_size=image_size, verbose=1)
    val_iter = data_iterator(val_id_type_list, batch_size=batch_size, image_size=image_size)

    total_counter = 0
    for X, Y_true in train_iter:
        total_counter += Y_true.shape[0]
        print("-- train", total_counter)
        if total_counter > samples_per_epoch:
            break

    total_counter = 0
    for X, Y_true in val_iter:
        total_counter += Y_true.shape[0]
        print("-- val", total_counter)
        if total_counter > nb_val_samples:
            break


def validate(val_id_type_list, batch_size=16, image_size=(224, 224)):
    val_iter = data_iterator(val_id_type_list, batch_size=batch_size, image_size=image_size, test_mode=True, verbose=1)

    total_counter = 0
    for X, Y_true, _ in val_iter:
        s = Y_true.shape[0]
        total_counter += s
        print("--", total_counter)

    if total_counter == 0:
        total_counter += 1


def predict(batch_size=16, image_size=(224, 224)):
    test_id_type_list = get_test_id_type_list()
    test_iter = data_iterator(test_id_type_list, batch_size=batch_size, image_size=image_size, test_mode=True, verbose=1)

    total_counter = 0
    for X, _, image_ids in test_iter:
        s = X.shape[0]
        total_counter += s
        print("--", total_counter)


if __name__ == "__main__":

    import platform

    batch_size = 16
    if 'c001' in platform.node():
        batch_size = 128
        print("-- On the cluster --")

    batch_size = 16
    if 'c001' in platform.node():
        batch_size = 128
        print("-- On the cluster --")

    print("\n {} - Get train/val lists ...".format(datetime.datetime.now()))
    train_id_type_list, val_id_type_list = get_trainval_id_type_lists()
    print("\n {} - Start training ...".format(datetime.datetime.now()))
    train(train_id_type_list, val_id_type_list, nb_epochs=1, batch_size=batch_size)
    print("\n {} - Start validation ...".format(datetime.datetime.now()))
    validate(val_id_type_list, batch_size=batch_size)
    print("\n {} - Start predictions and write submission ...".format(datetime.datetime.now()))
    predict(batch_size=batch_size)
    print("\n {} - Scripted finished".format(datetime.datetime.now()))
