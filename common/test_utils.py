import os
import datetime

import numpy as np
import pandas as pd
from keras import backend as K

# Project
from data_utils import test_ids, type_1_ids, type_2_ids, type_3_ids
from data_utils import GENERATED_DATA
from image_utils import imwrite
from xy_providers import cached_image_label_provider as xy_provider

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
                                  featurewise_std_normalization=True)

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
                         len(test_id_type_list),
                         batch_size=batch_size)

    total_counter = 0
    ll = len(test_id_type_list)
    for x, _, info in flow:
        y_pred = model.predict(x)
        s = y_pred.shape[0]
        for i in range(s):
            total_counter += 1
            print('-- %i / %i : %s, %s' % (total_counter, ll, info[i][0], info[i][1]))
            image_id = info[i][0] + '_' + info[i][1]
            imwrite(y_pred[i, :, :, :].transpose([1, 2, 0]), image_id, 'os_cervix_label__%s' % save_prefix)


# ###### Classification #######

def get_test_gen_flow(test_id_type_list,
                      normalize_data,
                      normalization,
                      save_prefix,
                      batch_size,
                      image_size,
                      option,
                      seed,
                      xy_provider_cache):

    xy_provider_verbose = 0

    if hasattr(K, 'image_data_format'):
        channels_first = K.image_data_format() == 'channels_first'
    elif hasattr(K, 'image_dim_ordering'):
        channels_first = K.image_dim_ordering() == 'th'
    else:
        raise Exception("Failed to find backend data format")

    test_gen = ImageDataGenerator(featurewise_center=normalize_data,
                                  featurewise_std_normalization=normalize_data,
                                  rotation_range=45.,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='constant')

    if normalize_data:
        if normalization == '':
            assert len(save_prefix) > 0, "WTF"
            # Load mean, std, principal_components if file exists
            filename = os.path.join(GENERATED_DATA, save_prefix + "_" + option + "_stats.npz")
            assert os.path.exists(filename), "WTF"
            print("Load existing file: %s" % filename)
            npzfile = np.load(filename)
            test_gen.mean = npzfile['mean']
            test_gen.std = npzfile['std']
        elif normalization == 'inception' or normalization == 'xception':
            # Preprocessing of Xception: keras/applications/xception.py
            print("Image normalization: ", normalization)
            test_gen.mean = 0.5
            test_gen.std = 0.5
        elif normalization == 'resnet' or normalization == 'vgg':
            print("Image normalization: ", normalization)
            test_gen.std = 1.0 / 255.0  # Rescale to [0.0, 255.0]
            m = np.array([123.68, 116.779, 103.939]) / 255.0 # RGB
            if channels_first:
                m = m[:, None, None]
            else:
                m = m[None, None, :]
            test_gen.mean = m

    test_flow = test_gen.flow(xy_provider(test_id_type_list,
                                          image_size=image_size,
                                          option=option,
                                          test_mode=True,
                                          seed=seed,
                                          with_labels=False,
                                          channels_first=channels_first,
                                          cache=xy_provider_cache,
                                          verbose=xy_provider_verbose),
                              # Ensure that all batches have the same size
                              len(test_id_type_list),
                              seed=seed,
                              batch_size=batch_size)

    return test_gen, test_flow


def classification_predict(model,
                           test_id_type_list,
                           option=None,
                           normalize_data=True,
                           normalization='',
                           image_size=(224, 224),
                           batch_size=16,
                           save_prefix="",
                           seed=2017,
                           verbose=1,
                           xy_provider_cache=None):

    if option == 'cervix/os':
        test_gen1, test_flow1 = get_test_gen_flow(test_id_type_list,
                                                  normalize_data,
                                                  normalization,
                                                  save_prefix,
                                                  batch_size,
                                                  image_size,
                                                  option='cervix',
                                                  seed=seed,
                                                  xy_provider_cache=xy_provider_cache)

        test_gen2, test_flow2 = get_test_gen_flow(test_id_type_list,
                                                  normalize_data,
                                                  normalization,
                                                  save_prefix,
                                                  batch_size,
                                                  image_size=tuple([int(s/2) for s in image_size]),
                                                  option='os',
                                                  seed=seed,
                                                  xy_provider_cache=xy_provider_cache)

        test_flow = map(lambda t: ([t[0][0], t[1][0]], t[0][1], t[0][2]), zip(test_flow1, test_flow2))
    else:
        test_gen, test_flow = get_test_gen_flow(test_id_type_list,
                                                normalize_data,
                                                normalization,
                                                save_prefix,
                                                batch_size,
                                                image_size,
                                                option,
                                                seed,
                                                xy_provider_cache)

    # flow = test_gen.flow(xy_provider(test_id_type_list,
    #                                  image_size=image_size,
    #                                  option=option,
    #                                  channels_first=channels_first,
    #                                  cache=xy_provider_cache),
    #                      # Ensure that all batches have the same size
    #                      len(test_id_type_list),
    #                      batch_size=batch_size)

    df = pd.DataFrame(columns=['image_name', 'Type_1', 'Type_2', 'Type_3'])
    total_counter = 0
    ll = len(test_id_type_list)

    for x, _, info in test_flow:
        y_pred = model.predict(x)
        s = y_pred.shape[0]
        if verbose > 0:
            print("--", total_counter, '/', ll)
        for i in range(s):
            df.loc[total_counter, :] = (info[i][0] + '.jpg',) + tuple(y_pred[i, :])
            total_counter += 1

    df = df.apply(pd.to_numeric, errors='ignore')
    return df
    # now = datetime.datetime.now()
    # sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    # sub_file = os.path.join('..', 'results', sub_file)
    # df.to_csv(sub_file, index=False)


