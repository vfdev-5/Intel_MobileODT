import os
import datetime

import numpy as np
import pandas as pd
from keras import backend as K

# Project
from data_utils import test_ids, type_to_index, type_1_ids, type_2_ids, type_3_ids
from data_utils import GENERATED_DATA
from image_utils import imwrite, get_image_data
from xy_providers import cached_image_provider as xy_provider

# Local keras-contrib:
from preprocessing.image.generators import ImageMaskGenerator, ImageDataGenerator


def get_test_id_type_list():
    return [(image_id, 'Test') for image_id in test_ids]


def get_test_id_type_list2(annotations):
    trainval_id_type_list = []
    for annotation in annotations:
        image_name = annotation['filename']
        image_id = os.path.basename(image_name)[:-4]
        image_type = os.path.split(os.path.dirname(image_name))[1]
        trainval_id_type_list.append((image_id, image_type))

    test_id_type_list = [(image_id, 'Test') for image_id in test_ids]
    type_ids=(type_1_ids, type_2_ids, type_3_ids)
    image_types = ["Type_1", "Type_2", "Type_3"]

    for image_ids, image_type in zip(type_ids, image_types):
        for image_id in image_ids:
            if (image_id, image_type) not in trainval_id_type_list:
                test_id_type_list.append((image_id, image_type))
    return test_id_type_list


# ###### Segmentation #######

def segmentation_predict(model,
                         test_id_type_list,
                         save_prefix,
                         batch_size=16,
                         xy_provider_cache=None,
                         image_size=(224, 224)):

    test_gen = ImageMaskGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True,
                                  rotation_range=90.,
                                  horizontal_flip=True,
                                  vertical_flip=True)

    assert len(save_prefix) > 0, "WTF"
    # Load mean, std, principal_components if file exists
    filename = os.path.join(GENERATED_DATA, save_prefix + "_stats.npz")
    assert os.path.exists(filename), "WTF"
    print("Load existing file: %s" % filename)
    npzfile = np.load(filename)
    test_gen.mean = npzfile['mean']
    test_gen.std = npzfile['std']

    flow = test_gen.flow(xy_provider(test_id_type_list,
                                     image_size=image_size,
                                     cache=xy_provider_cache),
                         # Ensure that all batches have the same size
                         len(test_id_type_list),
                         batch_size=batch_size)

    total_counter = 0
    for x, _, info in flow:
        y_pred = model.predict(x)
        s = y_pred.shape[0]
        for i in range(s):
            total_counter += 1
            print("--", total_counter, info[i])
            imwrite(y_pred[i, :, :, :], info[i][0] + '_' + info[i][1], 'pred_label')


# ###### Classification #######

def classification_predict(model,
                           test_id_type_list,
                           batch_size=16,
                           save_prefix="",
                           info='',
                           xy_provider_cache=None):

    normalize_data = True
    image_size = (299, 299)

    if hasattr(K, 'image_data_format'):
        channels_first = K.image_data_format() == 'channels_first'
    elif hasattr(K, 'image_dim_ordering'):
        channels_first = K.image_dim_ordering() == 'th'
    else:
        raise Exception("Failed to find backend data format")

    test_gen = ImageDataGenerator(featurewise_center=normalize_data,
                                  featurewise_std_normalization=normalize_data)

    if normalize_data:
        assert len(save_prefix) > 0, "WTF"
        # Load mean, std, principal_components if file exists
        filename = os.path.join(GENERATED_DATA, save_prefix + "_stats.npz")
        assert os.path.exists(filename), "WTF"
        print("Load existing file: %s" % filename)
        npzfile = np.load(filename)
        test_gen.mean = npzfile['mean']
        test_gen.std = npzfile['std']

    flow = test_gen.flow(xy_provider(test_id_type_list,
                                     image_size=image_size,
                                     channels_first=channels_first,
                                     cache=xy_provider_cache),
                         # Ensure that all batches have the same size
                         len(test_id_type_list),
                         batch_size=batch_size)

    df = pd.DataFrame(columns=['image_name', 'Type_1', 'Type_2', 'Type_3'])
    total_counter = 0
    ll = len(test_id_type_list)
    for x, _, image_ids in flow:
        y_pred = model.predict(x)
        s = x.shape[0]
        print("--", total_counter, '/', ll)
        for i in range(s):
            df.loc[total_counter, :] = (image_ids[i][0] + '.jpg',) + tuple(y_pred[i, :])
            total_counter += 1

    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    sub_file = os.path.join('..', 'results', sub_file)
    df.to_csv(sub_file, index=False)