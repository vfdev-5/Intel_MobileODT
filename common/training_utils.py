
import os
import numpy as np
import cv2
import sys
from datetime import datetime

if sys.version_info < (3,):
    from itertools import izip as zip, imap as map

from keras.preprocessing.image import random_rotation, random_shift, flip_axis
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras import backend as K
from keras import __version__ as keras_version

from imgaug.imgaug import augmenters as iaa

# Project
from data_utils import type_to_index, type_1_ids, type_2_ids, type_3_ids
from data_utils import additional_type_1_ids, additional_type_2_ids, additional_type_3_ids
from data_utils import GENERATED_DATA, get_id_type_list_from_annotations
from image_utils import get_image_data
from metrics import jaccard_index, logloss_mc
from xy_providers import cached_image_mask_provider, cached_image_label_provider


# Local keras-contrib:
from preprocessing.image.generators import ImageMaskGenerator, ImageDataGenerator


def find_best_weights_file(weights_files, start_index=None, end_index=-3):
    best_val_loss = 1e5
    best_weights_filename = ""
    for f in weights_files:
        if start_index is None:
            start_index = os.path.basename(f).index('-')
        loss_str = os.path.basename(f)[start_index+1:end_index]
        if '-' in loss_str:
            end_index = loss_str.index('-')
            loss_str = loss_str[:end_index]
        loss = float(loss_str)
        if best_val_loss > loss:
            best_val_loss = loss
            best_weights_filename = f
    return best_weights_filename, best_val_loss


def find_best_weights_file2(weights_files, field_name='val_loss', best_min=True):

    best_value = 1e5 if best_min else -1e5
    if best_min:
        best_value = 1e5
        comp = lambda a, b: a > b
    else:
        best_value = -1e5
        comp = lambda a, b: a < b

    if '=' != field_name[-1]:
        field_name += '='

    best_weights_filename = ""
    for f in weights_files:
        index = f.find(field_name)
        index += len(field_name)
        assert index >= 0, "Field name '%s' is not found in '%s'" % (field_name, f)
        end = f.find('_', index)
        val = float(f[index:end])
        if comp(best_value, val):
            best_value = val
            best_weights_filename = f
    return best_weights_filename, best_value


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
    val_ids = [ids[tl:tl + vl] for ids, tl, vl in zip(type_ids, train_ll, val_ll)]
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


def get_trainval_id_type_lists3(n_images_per_class=730, val_split=0.3, seed=2017):

    np.random.seed(seed)

    def _get_id_type_list(n_images, type_ids, image_types):
        id_type_list = []
        for ids, image_type in zip(type_ids, image_types):
            for image_id in ids:
                id_type_list.append((image_id, image_type))

        assert len(id_type_list) > n_images, "WTF"
        id_type_list = id_type_list[:n_images]
        np.random.shuffle(id_type_list)
        return id_type_list

    id_type_1_list = _get_id_type_list(n_images_per_class,
                                       [type_1_ids, additional_type_1_ids],
                                       ["Type_1", "AType_1"])

    id_type_2_list = _get_id_type_list(n_images_per_class,
                                       [type_2_ids, additional_type_2_ids],
                                       ["Type_2", "AType_2"])

    id_type_3_list = _get_id_type_list(n_images_per_class,
                                       [type_3_ids, additional_type_3_ids],
                                       ["Type_3", "AType_3"])

    #print(len(id_type_1_list), len(id_type_2_list), len(id_type_3_list))
    #print(id_type_1_list[:2], id_type_2_list[:2], id_type_3_list[:2])

    train_ll = int(n_images_per_class * (1.0 - val_split))
    train_id_type_list = list(id_type_1_list[:train_ll])
    train_id_type_list.extend(id_type_2_list[:train_ll])
    train_id_type_list.extend(id_type_3_list[:train_ll])

    val_id_type_list = list(id_type_1_list[train_ll:])
    val_id_type_list.extend(id_type_2_list[train_ll:])
    val_id_type_list.extend(id_type_3_list[train_ll:])

    np.random.shuffle(train_id_type_list)
    np.random.shuffle(val_id_type_list)

    return train_id_type_list, val_id_type_list


def get_trainval_id_type_lists2(annotations, val_split=0.3):
    n = len(annotations)
    np.random.shuffle(annotations)
    ll = int(n * (1.0 - val_split))
    train_annotations = annotations[:ll]
    val_annotations = annotations[ll:]

    train_id_type_list = get_id_type_list_from_annotations(train_annotations)
    val_id_type_list = get_id_type_list_from_annotations(val_annotations)

    return train_id_type_list, val_id_type_list


def get_all_trainval_id_type_list(n_images_per_class=1400):
    # Number of train images per one class

    def get_id_type_list(n_images, type_ids, image_types):
        id_type_list = []
        for ids, image_type in zip(type_ids, image_types):
            for image_id in ids:
                id_type_list.append((image_id, image_type))

        assert len(id_type_list) > n_images, "WTF: %i, %i" % (len(id_type_list), n_images)
        return id_type_list[:n_images] if n_images > 0 else id_type_list

    id_type_1_list = get_id_type_list(n_images_per_class,
                                      [type_1_ids, additional_type_1_ids],
                                      ["Type_1", "AType_1"])

    id_type_2_list = get_id_type_list(n_images_per_class,
                                      [type_2_ids, additional_type_2_ids],
                                      ["Type_2", "AType_2"])

    id_type_3_list = get_id_type_list(n_images_per_class,
                                      [type_3_ids, additional_type_3_ids],
                                      ["Type_3", "AType_3"])

    all_trainval_id_type_list = list(id_type_1_list)
    all_trainval_id_type_list.extend(id_type_2_list)
    all_trainval_id_type_list.extend(id_type_3_list)
    return all_trainval_id_type_list


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


def exp_decay(epoch, lr=1e-3, a=0.925, init_epoch=0):
    return lr * np.exp(-(1.0 - a) * (epoch + init_epoch))


def step_decay(epoch, lr=1e-3, base=2.0, period=50, init_epoch=0):
    return lr * base ** (-np.floor((epoch + init_epoch) * 1.0 / period))


def random_rgb_to_green_generic(*args):
    assert len(args) > 0, "List of arguments should not be empty"
    output_args = list(args)
    if np.random.rand() > 0.90:
        for i, arg in enumerate(output_args):
            output_args[i] = arg.copy()
            output_args[i][0, :, :] = 0  # Red -> 0
            output_args[i][2, :, :] = 0  # Blue -> 0
    return output_args[0] if len(output_args) == 1 else output_args


def random_rgb_to_green(x, y):

    if np.random.rand() > 0.90:
        out = x.copy()
        out[0, :, :] = 0  # Red -> 0
        out[2, :, :] = 0  # Blue -> 0
    else:
        out = x
    return out, y


def random_more_blue(x, y, channels_first):

    def more_blue(img, b=0, channels_first=False):
        out = img.copy()
        f1 = np.random.rand() * 0.05 + 0.6
        f2 = np.random.rand() * 0.05 + 0.7
        if channels_first:
            out[2, :, :] = np.clip(img[2, :, :] * 1.15 + b, 0, 1.0)
            out[0, :, :] = np.clip(img[0, :, :] * f1 + b, 0, 1.0)
            out[1, :, :] = np.clip(img[1, :, :] * f2 + b, 0, 1.0)
        else:
            out[:, :, 2] = np.clip(img[:, :, 2] * 1.15 + b, 0, 1.0)
            out[:, :, 0] = np.clip(img[:, :, 0] * f1 + b, 0, 1.0)
            out[:, :, 1] = np.clip(img[:, :, 1] * f2 + b, 0, 1.0)
        return out

    return more_blue(x, b=np.random.randint(-30, 20) * 1.0/255.0, channels_first=channels_first), y


def random_more_yellow(x, y, channels_first):

    def more_yellow(img, b=0, channels_first=False):
        out = img.copy()
        f1 = np.random.rand() * 0.15 + 0.6
        f2 = np.random.rand() * 0.05 + 0.15
        if channels_first:
            out[0, :, :] = np.clip(img[0, :, :] * 1.05 + b, 0, 1.0)
            out[1, :, :] = np.clip(img[1, :, :] * f1 + b, 0, 1.0)
            out[2, :, :] = np.clip(img[2, :, :] * f2 + b, 0, 1.0)
        else:
            out[:, :, 0] = np.clip(img[:, :, 0] * 1.05 + b, 0, 1.0)
            out[:, :, 1] = np.clip(img[:, :, 1] * f1 + b, 0, 1.0)
            out[:, :, 2] = np.clip(img[:, :, 2] * f2 + b, 0, 1.0)
        return out

    return more_yellow(x, b=np.random.randint(-20, 50) * 1.0/255.0, channels_first=channels_first), y


def random_inversion(x, y=None):
    r = np.random.rand()
    if r > 0.5:
        return np.power(1.0 - x, 2.0)
    else:
        return x


# ###### Segmentation task #######


def segmentation_train(model,
                       train_id_type_list,
                       val_id_type_list,
                       batch_size=16,
                       nb_epochs=10,
                       lrate_decay_f=None,
                       image_size=(224, 224),
                       samples_per_epoch=2048,
                       nb_val_samples=1024,
                       xy_provider_cache=None,
                       save_prefix="",
                       seed=None,
                       verbose=1):

    samples_per_epoch = (samples_per_epoch // batch_size + 1) * batch_size
    nb_val_samples = (nb_val_samples // batch_size + 1) * batch_size

    if not os.path.exists('weights'):
        os.mkdir('weights')

    weights_filename = os.path.join("weights", save_prefix +
                                    "_{epoch:02d}_val_loss={val_loss:.4f}_val_jaccard_index={val_jaccard_index:.4f}.h5")
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True)
    callbacks = [model_checkpoint, ]
    if lrate_decay_f is not None:
        lrate = LearningRateScheduler(lrate_decay_f)
        callbacks.append(lrate)

    print("\n-- Training parameters: %i, %i, %i, %i" % (batch_size, nb_epochs, samples_per_epoch, nb_val_samples))

    xy_provider = cached_image_mask_provider
    xy_provider_verbose = 0
    xy_provider_label_type = 'trainval_label_0'

    if hasattr(K, 'image_data_format'):
        channels_first = K.image_data_format() == 'channels_first'
    elif hasattr(K, 'image_dim_ordering'):
        channels_first = K.image_dim_ordering() == 'th'
    else:
        raise Exception("Failed to find backend data format")

    r1 = lambda x, y: random_more_blue(x, y, channels_first)
    r2 = lambda x, y: random_more_yellow(x, y, channels_first)

    def random_blue_or_yellow(x, y):
        r = np.random.rand()
        if r > 0.667:
            return r1(x, y)
        elif 0.333 < r <= 0.667:
            return r2(x, y)
        else:
            return x, y

    train_gen = ImageMaskGenerator(pipeline=('random_transform', random_blue_or_yellow, 'standardize'),
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=45.,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=[0.65, 1.2],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='constant')

    val_gen = ImageMaskGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=45.,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='constant')

    print("\n-- Fit stats of train dataset")
    train_gen.fit(xy_provider(train_id_type_list,
                              mask_type=xy_provider_label_type,
                              test_mode=True,
                              cache=xy_provider_cache,
                              image_size=image_size),
                  len(train_id_type_list),
                  augment=True,
                  save_to_dir=GENERATED_DATA,
                  save_prefix=save_prefix,
                  batch_size=4,
                  seed=seed,
                  verbose=verbose)

    val_gen.mean = train_gen.mean
    val_gen.std = train_gen.std
    val_gen.principal_components = train_gen.principal_components

    print("\n-- Fit model")
    try:

        train_flow = train_gen.flow(xy_provider(train_id_type_list,
                                                mask_type=xy_provider_label_type,
                                                image_size=image_size,
                                                cache=xy_provider_cache,
                                                verbose=xy_provider_verbose),
                                    # Ensure that all batches have the same size
                                    (len(train_id_type_list) // batch_size) * batch_size,
                                    seed=seed,
                                    batch_size=batch_size)

        val_flow = val_gen.flow(xy_provider(val_id_type_list,
                                            mask_type=xy_provider_label_type,
                                            image_size=image_size,
                                            cache=xy_provider_cache,
                                            verbose=xy_provider_verbose),
                                # Ensure that all batches have the same size
                                (len(val_id_type_list) // batch_size) * batch_size,
                                seed=seed,
                                batch_size=batch_size)

        history = model.fit_generator(train_flow,
                                      samples_per_epoch=samples_per_epoch,
                                      nb_epoch=nb_epochs,
                                      validation_data=val_flow,
                                      nb_val_samples=nb_val_samples,
                                      callbacks=callbacks,
                                      verbose=verbose)

        # save the last
        val_loss = history.history['val_loss'][-1]
        val_jaccard_index = history.history['val_jaccard_index'][-1]
        weights_filename = weights_filename.format(epoch=nb_epochs, val_loss=val_loss, val_jaccard_index=val_jaccard_index)
        model.save_weights(weights_filename)

        return history

    except KeyboardInterrupt:
        pass


def segmentation_validate(model,
                          val_id_type_list,
                          save_prefix,
                          batch_size=16,
                          xy_provider_cache=None,
                          image_size=(224, 224)):

    xy_provider = cached_image_mask_provider
    xy_provider_label_type = 'trainval_label_0'

    val_gen = ImageMaskGenerator(featurewise_center=True,
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
    val_gen.mean = npzfile['mean']
    val_gen.std = npzfile['std']

    flow = val_gen.flow(xy_provider(val_id_type_list,
                                    mask_type=xy_provider_label_type,
                                    test_mode=True,
                                    cache=xy_provider_cache,
                                    image_size=image_size),
                        # Ensure that all batches have the same size
                        len(val_id_type_list),
                        batch_size=batch_size)

    mean_jaccard_index = 0.0
    total_counter = 0
    for x, y_true, info in flow:
        s = y_true.shape[0]
        total_counter += s
        y_pred = model.predict(x)
        ji = jaccard_index(y_true, y_pred)
        mean_jaccard_index += s * ji
        print("--", total_counter, "batch jaccard index : ", ji, " | info:", info)

    if total_counter == 0:
        total_counter += 1

    mean_jaccard_index *= 1.0 / total_counter
    print("Mean jaccard index : ", mean_jaccard_index)

# ##########################################################################################
# ######################### Classification #################################################
# ##########################################################################################


def random_imgaug(x, seq):
    return seq.augment_images([x, ])[0]


def get_train_gen_flow(train_id_type_list,
                       normalize_data,
                       normalization,
                       batch_size,
                       option,
                       image_size,
                       seed,
                       save_prefix,
                       xy_provider_cache,
                       verbose):

    xy_provider = cached_image_label_provider
    xy_provider_verbose = 0

    if hasattr(K, 'image_data_format'):
        channels_first = K.image_data_format() == 'channels_first'
    elif hasattr(K, 'image_dim_ordering'):
        channels_first = K.image_dim_ordering() == 'th'
    else:
        raise Exception("Failed to find backend data format")

    # r1 = lambda x: random_more_blue(x, None, channels_first)[0]
    # r2 = lambda x: random_more_yellow(x, None, channels_first)[0]
    #
    # def _random_blue_or_yellow(x):
    #
    #     r = np.random.rand()
    #     if r > 0.75:
    #         return r1(x)
    #     elif 0.5 < r <= 0.75:
    #         return r2(x)
    #     else:
    #         return x

    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.15, 0.6))),
        iaa.Sometimes(0.5, iaa.Sharpen(alpha=0.9, lightness=(0.3, 1.4))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.02*255), per_channel=True)),
        iaa.Sometimes(0.5, iaa.ContrastNormalization(alpha=(0.75, 1.3))),
        iaa.Sometimes(0.75, iaa.PiecewiseAffine(scale=(0.001, 0.05), mode='reflect')),
        iaa.Sometimes(0.75, iaa.Affine(translate_px=(-10, 10),
                                       rotate=(-90.0, 90.0),
                                       mode='reflect')),
        iaa.Sometimes(0.75, iaa.Add(value=(-55, 15), per_channel=True)),
    ],
        random_order=True,
        random_state=seed)

    def _random_imgaug(x):
        return random_imgaug(255.0 * x, seq) * 1.0/255.0

    train_gen = ImageDataGenerator(pipeline=(#'random_transform',
                                             # _random_blue_or_yellow,
                                             _random_imgaug,
                                             'standardize'),
                                   featurewise_center=normalize_data,
                                   featurewise_std_normalization=normalize_data,
                                   rotation_range=45.,
                                   width_shift_range=0.025, height_shift_range=0.025,
                                   zoom_range=[0.85, 1.05],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='reflect')

    if normalize_data:
        if normalization == '':
            print("\n-- Fit stats of train dataset")
            train_gen.fit(xy_provider(train_id_type_list,
                                      image_size=image_size,
                                      option=option,
                                      test_mode=True,
                                      channels_first=channels_first,
                                      cache=xy_provider_cache),
                          len(train_id_type_list),
                          augment=True,
                          seed=seed,
                          save_to_dir=GENERATED_DATA,
                          save_prefix=save_prefix + '_' + option,
                          batch_size=4,
                          verbose=verbose)
        elif normalization == 'inception' or normalization == 'xception':
            # Preprocessing of Xception: keras/applications/xception.py
            print("Image normalization: ", normalization)
            train_gen.mean = 0.5
            train_gen.std = 0.5
        elif normalization == 'resnet' or normalization == 'vgg':
            print("Image normalization: ", normalization)
            train_gen.std = 1.0 / 255.0  # Rescale to [0.0, 255.0]
            m = np.array([123.68, 116.779, 103.939]) / 255.0  # RGB
            if channels_first:
                m = m[:, None, None]
            else:
                m = m[None, None, :]
            train_gen.mean = m

    train_flow = train_gen.flow(xy_provider(train_id_type_list,
                                            image_size=image_size,
                                            option=option,
                                            seed=seed,
                                            channels_first=channels_first,
                                            cache=xy_provider_cache,
                                            verbose=xy_provider_verbose),
                                # Ensure that all batches have the same size
                                (len(train_id_type_list) // batch_size) * batch_size,
                                seed=seed,
                                batch_size=batch_size)

    return train_gen, train_flow


def get_val_gen_flow(val_id_type_list,
                     normalize_data,
                     normalization,
                     save_prefix,
                     batch_size,
                     image_size,
                     option,
                     seed,
                     xy_provider_cache,
                     test_mode=False):

    xy_provider = cached_image_label_provider
    xy_provider_verbose = 0

    if hasattr(K, 'image_data_format'):
        channels_first = K.image_data_format() == 'channels_first'
    elif hasattr(K, 'image_dim_ordering'):
        channels_first = K.image_dim_ordering() == 'th'
    else:
        raise Exception("Failed to find backend data format")

    # r1 = lambda x: random_more_blue(x, None, channels_first)[0]
    # r2 = lambda x: random_more_yellow(x, None, channels_first)[0]
    #
    # def random_blue_or_yellow(x):
    #     r = np.random.rand()
    #     if r > 0.75:
    #         return r1(x)
    #     elif 0.5 < r <= 0.75:
    #         return r2(x)
    #     else:
    #         return x

    val_gen = ImageDataGenerator(pipeline=('random_transform', 'standardize'),
                                 featurewise_center=normalize_data,
                                 featurewise_std_normalization=normalize_data,
                                 rotation_range=45.,
                                 width_shift_range=0.025, height_shift_range=0.025,
                                 zoom_range=[0.85, 1.05],
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='reflect')

    if normalize_data:
        if normalization == '':
            assert len(save_prefix) > 0, "WTF"
            # Load mean, std, principal_components if file exists
            filename = os.path.join(GENERATED_DATA, save_prefix + "_" + option + "_stats.npz")
            assert os.path.exists(filename), "WTF"
            print("Load existing file: %s" % filename)
            npzfile = np.load(filename)
            val_gen.mean = npzfile['mean']
            val_gen.std = npzfile['std']
        elif normalization == 'inception' or normalization == 'xception':
            # Preprocessing of Xception: keras/applications/xception.py
            print("Image normalization: ", normalization)
            val_gen.mean = 0.5
            val_gen.std = 0.5
        elif normalization == 'resnet' or normalization == 'vgg':
            print("Image normalization: ", normalization)
            val_gen.std = 1.0 / 255.0  # Rescale to [0.0, 255.0]
            m = np.array([123.68, 116.779, 103.939]) / 255.0 # RGB
            if channels_first:
                m = m[:, None, None]
            else:
                m = m[None, None, :]
            val_gen.mean = m

    val_flow = val_gen.flow(xy_provider(val_id_type_list,
                                        image_size=image_size,
                                        option=option,
                                        seed=seed,
                                        test_mode=test_mode,
                                        channels_first=channels_first,
                                        cache=xy_provider_cache,
                                        verbose=xy_provider_verbose),
                            # Ensure that all batches have the same size
                            (len(val_id_type_list) // batch_size) * batch_size,
                            seed=seed,
                            batch_size=batch_size)

    return val_gen, val_flow


def classification_train(model,
                         train_id_type_list,
                         val_id_type_list,
                         option=None,
                         normalize_data=True,
                         normalization='',
                         batch_size=16,
                         nb_epochs=10,
                         image_size=(224, 224),
                         lrate_decay_f=None,
                         samples_per_epoch=2048,
                         nb_val_samples=1024,
                         xy_provider_cache=None,
                         class_weight={},
                         seed=None,
                         save_prefix="",
                         verbose=1):

    samples_per_epoch = (samples_per_epoch // batch_size + 1) * batch_size
    nb_val_samples = (nb_val_samples // batch_size + 1) * batch_size

    if not os.path.exists('weights'):
        os.mkdir('weights')

    weights_filename = os.path.join("weights", save_prefix +
                                    "_{epoch:02d}_val_loss={val_loss:.4f}_" +
                                    "val_cat_crossentropy={val_categorical_crossentropy:.4f}_" +
                                    "val_cat_accuracy={val_categorical_accuracy:.4f}.h5")
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss',
                                       save_best_only=True, save_weights_only=True)
    now = datetime.now()
    csv_logger = CSVLogger('weights/training_%s_%s.log' % (save_prefix, str(now.strftime("%Y-%m-%d-%H-%M"))))
    callbacks = [model_checkpoint, csv_logger, ]
    if lrate_decay_f is not None:
        lrate = LearningRateScheduler(lrate_decay_f)
        callbacks.append(lrate)

    print("\n-- Training parameters: %i, %i, %i, %i" % (batch_size, nb_epochs, samples_per_epoch, nb_val_samples))
    print("\n-- Fit model")
    try:
        if option == 'cervix/os':
            train_gen1, train_flow1 = get_train_gen_flow(train_id_type_list=train_id_type_list,
                                                         normalize_data=normalize_data,
                                                         normalization=normalization,
                                                         batch_size=batch_size,
                                                         seed=seed,
                                                         image_size=image_size,
                                                         option='cervix',
                                                         save_prefix=save_prefix,
                                                         xy_provider_cache=xy_provider_cache,
                                                         verbose=verbose)

            val_gen1, val_flow1 = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                                   normalization=normalization,
                                                   save_prefix=save_prefix,
                                                   normalize_data=normalize_data,
                                                   batch_size=batch_size,
                                                   seed=seed,
                                                   image_size=image_size,
                                                   option='cervix',
                                                   xy_provider_cache=xy_provider_cache)

            train_gen2, train_flow2 = get_train_gen_flow(train_id_type_list=train_id_type_list,
                                                         normalize_data=normalize_data,
                                                         normalization=normalization,
                                                         batch_size=batch_size,
                                                         seed=seed,
                                                         image_size=tuple([int(s/2) for s in image_size]),
                                                         option='os',
                                                         save_prefix=save_prefix,
                                                         xy_provider_cache=xy_provider_cache,
                                                         verbose=verbose)

            val_gen2, val_flow2 = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                                   normalization=normalization,
                                                   save_prefix=save_prefix,
                                                   normalize_data=normalize_data,
                                                   batch_size=batch_size,
                                                   seed=seed,
                                                   image_size=tuple([int(s/2) for s in image_size]),
                                                   option='os',
                                                   xy_provider_cache=xy_provider_cache)

            train_flow = map(lambda t: ([t[0][0], t[1][0]], t[0][1]), zip(train_flow1, train_flow2))
            val_flow = map(lambda t: ([t[0][0], t[1][0]], t[0][1]), zip(val_flow1, val_flow2))
        else:
            train_gen, train_flow = get_train_gen_flow(train_id_type_list=train_id_type_list,
                                                       normalize_data=normalize_data,
                                                       normalization=normalization,
                                                       batch_size=batch_size,
                                                       seed=seed,
                                                       image_size=image_size,
                                                       option=option,
                                                       save_prefix=save_prefix,
                                                       xy_provider_cache=xy_provider_cache,
                                                       verbose=verbose)

            val_gen, val_flow = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                                 normalization=normalization,
                                                 save_prefix=save_prefix,
                                                 normalize_data=normalize_data,
                                                 batch_size=batch_size,
                                                 seed=seed,
                                                 image_size=image_size,
                                                 option=option,
                                                 xy_provider_cache=xy_provider_cache)

        # New or old Keras API
        if int(keras_version[0]) == 2:
            print("- New Keras API found -")
            history = model.fit_generator(generator=train_flow,
                                          steps_per_epoch=(samples_per_epoch // batch_size),
                                          epochs=nb_epochs,
                                          validation_data=val_flow,
                                          validation_steps=(nb_val_samples // batch_size),
                                          callbacks=callbacks,
                                          class_weight=class_weight,
                                          verbose=verbose)
        else:
            history = model.fit_generator(generator=train_flow,
                                          samples_per_epoch=samples_per_epoch,
                                          nb_epoch=nb_epochs,
                                          validation_data=val_flow,
                                          nb_val_samples=nb_val_samples,
                                          callbacks=callbacks,
                                          class_weight=class_weight,
                                          verbose=verbose)
        # save the last
        #val_loss = history.history['val_loss'][-1]
        #val_acc = history.history['val_acc'][-1]
        #val_recall = history.history['val_recall'][-1]
        #val_precision = history.history['val_precision'][-1]
        #weights_filename = weights_filename.format(epoch=nb_epochs,
        #                                           val_loss=val_loss,
        #                                           val_acc=val_acc,
        #                                           val_precision=val_precision,
        #                                           val_recall=val_recall)

        val_loss = history.history['val_loss'][-1]
        val_cat_crossentropy = history.history['val_categorical_crossentropy'][-1]
        val_cat_accuracy = history.history['val_categorical_accuracy'][-1]
        weights_filename = weights_filename.format(epoch=nb_epochs,
                                                   val_loss=val_loss,
                                                   val_categorical_crossentropy=val_cat_crossentropy,
                                                   val_categorical_accuracy=val_cat_accuracy)

        model.save_weights(weights_filename)
        return history

    except KeyboardInterrupt:
        pass


def classification_validate(model,
                            val_id_type_list,
                            option=None,
                            normalize_data=True,
                            normalization='',
                            image_size=(224, 224),
                            save_prefix="",
                            batch_size=16,
                            seed=None,
                            xy_provider_cache=None,
                            verbose=1):

    if option == 'cervix/os':
        val_gen1, val_flow1 = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                               normalization=normalization,
                                               save_prefix=save_prefix,
                                               normalize_data=normalize_data,
                                               batch_size=batch_size,
                                               seed=seed,
                                               image_size=image_size,
                                               option='cervix',
                                               test_mode=True,
                                               xy_provider_cache=xy_provider_cache)

        val_gen2, val_flow2 = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                               normalization=normalization,
                                               save_prefix=save_prefix,
                                               normalize_data=normalize_data,
                                               batch_size=batch_size,
                                               seed=seed,
                                               image_size=tuple([int(s/2) for s in image_size]),
                                               option='os',
                                               test_mode=True,
                                               xy_provider_cache=xy_provider_cache)

        val_flow = map(lambda t: ([t[0][0], t[1][0]], t[0][1], t[0][2]), zip(val_flow1, val_flow2))
    else:
        val_gen, val_flow = get_val_gen_flow(val_id_type_list=val_id_type_list,
                                             normalization=normalization,
                                             save_prefix=save_prefix,
                                             normalize_data=normalize_data,
                                             batch_size=batch_size,
                                             seed=seed,
                                             image_size=image_size,
                                             option=option,
                                             test_mode=True,
                                             xy_provider_cache=xy_provider_cache)

    total_loss = 0.0
    total_counter = 0
    for x, y_true, info in val_flow:
        s = y_true.shape[0]
        total_counter += s
        y_pred = model.predict(x)
        loss = logloss_mc(y_true, y_pred)
        total_loss += s * loss
        if verbose > 1:
            print("--", total_counter, "batch loss : ", loss, " | info:", info)

    if total_counter == 0:
        total_counter += 1

    total_loss *= 1.0 / total_counter
    if verbose > 0:
        print("Total loss : ", total_loss)
    return total_loss


# ###### OLD #####


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

