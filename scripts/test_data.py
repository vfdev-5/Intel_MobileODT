
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


print("\n=========================")
print("Training dataset: ")
print("- type 1: ", len(type_1_ids))
print("- type 2: ", len(type_2_ids))
print("- type 3: ", len(type_3_ids))

print("Test dataset: ")
print("- ", len(test_ids))
print("=========================\n")


def read(type_ids, image_type):
    ll = len(type_ids)
    for i, image_id in enumerate(type_ids):
        print("----", image_id, image_type, i, "/", ll)
        img = get_image_data(image_id, image_type)
        print("-----", img.shape, img.min(), img.max(), img.mean(), img.std())


if __name__ == "__main__":

    print("\n {} - Read train data ...".format(datetime.datetime.now()))
    print("\n {} --- type 1 ".format(datetime.datetime.now()))
    # read(type_1_ids, "Type_1")

    print("\n {} --- type 2 ".format(datetime.datetime.now()))
    # read(type_2_ids, "Type_2")

    print("\n {} --- type 3 ".format(datetime.datetime.now()))
    read(type_3_ids, "Type_3")

    print("\n {} - Read test data ...".format(datetime.datetime.now()))
    read(test_ids, "Test")
    print("\n {} - Scripted finished".format(datetime.datetime.now()))
