
import os
import numpy as np
import cv2

from keras.preprocessing.image import random_rotation, random_shift, flip_axis
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

# Project
from data_utils import test_ids, type_to_index, type_1_ids, type_2_ids, type_3_ids
from data_utils import GENERATED_DATA
from image_utils import get_image_data, imwrite
from metrics import jaccard_index, logloss_mc
from xy_providers import cached_image_mask_provider, cached_image_label_provider


# Local keras-contrib:
from preprocessing.image.generators import ImageMaskGenerator, ImageDataGenerator
from preprocessing.image.iterators import ImageMaskIterator, ImageDataIterator


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


def get_trainval_id_type_lists2(annotations, val_split=0.3):
    n = len(annotations)
    np.random.shuffle(annotations)
    ll = int(n * (1.0 - val_split))
    train_annotations = annotations[:ll]
    val_annotations = annotations[ll:]

    train_id_type_list = []
    for i, annotation in enumerate(train_annotations):
        image_name = annotation['filename']
        image_id = os.path.basename(image_name)[:-4]
        image_type = os.path.split(os.path.dirname(image_name))[1]
        train_id_type_list.append((image_id, image_type))

    val_id_type_list = []
    for i, annotation in enumerate(val_annotations):
        image_name = annotation['filename']
        image_id = os.path.basename(image_name)[:-4]
        image_type = os.path.split(os.path.dirname(image_name))[1]
        val_id_type_list.append((image_id, image_type))

    return train_id_type_list, val_id_type_list


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


def exp_decay(epoch, lr=1e-3, a=0.925):
    return lr * np.exp(-(1.0 - a) * epoch)


def random_rgb_to_green(*args):
    assert len(args) > 0, "List of arguments should not be empty"
    output_args = list(args)
    if np.random.rand() > 0.75:
        for i, arg in enumerate(output_args):
            output_args[i] = arg.copy()
            output_args[i][0, :, :] = 0  # Red -> 0
            output_args[i][2, :, :] = 0  # Blue -> 0
    return output_args[0] if len(output_args) == 1 else output_args


# ###### Segmentation task #######


def segmentation_train(model,
                       train_id_type_list,
                       val_id_type_list,
                       batch_size=16,
                       nb_epochs=10,
                       lrate_decay_f=exp_decay,
                       image_size=(224, 224),
                       samples_per_epoch=1024,
                       nb_val_samples=256,
                       xy_provider_cache=None,
                       save_prefix="",
                       seed=None,
                       verbose=1):

    samples_per_epoch = (samples_per_epoch // batch_size) * batch_size
    nb_val_samples = (nb_val_samples // batch_size) * batch_size

    if not os.path.exists('weights'):
        os.mkdir('weights')

    weights_filename = os.path.join("weights", save_prefix + "_{epoch:02d}-{val_loss:.4f}.h5")
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True)
    callbacks = [model_checkpoint, ]
    if lrate_decay_f is not None:
        lrate = LearningRateScheduler(lrate_decay_f)
        callbacks.append(lrate)

    print("-- Training parameters: ", batch_size, nb_epochs, samples_per_epoch, nb_val_samples)

    xy_provider = cached_image_mask_provider
    xy_provider_verbose = 0
    xy_provider_label_type = 'trainval_label_0'

    train_gen = ImageMaskGenerator(pipeline=('random_transform', random_rgb_to_green, 'standardize'),
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=90.,
                                   width_shift_range=0.15, height_shift_range=0.15,
                                   shear_range=3.14/6.0,
                                   zoom_range=0.25,
                                   channel_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True)
    val_gen = ImageMaskGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=90.,
                                 horizontal_flip=True,
                                 vertical_flip=True)

    print("-- Fit stats of train dataset")
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

    print("-- Fit model")
    history = model.fit_generator(
        train_gen.flow(xy_provider(train_id_type_list,
                                   mask_type=xy_provider_label_type,
                                   image_size=image_size,
                                   cache=xy_provider_cache,
                                   verbose=xy_provider_verbose),
                       # Ensure that all batches have the same size
                       (len(train_id_type_list) // batch_size) * batch_size,
                       seed=seed,
                       batch_size=batch_size),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epochs,
        validation_data=val_gen.flow(xy_provider(val_id_type_list,
                                                 mask_type=xy_provider_label_type,
                                                 image_size=image_size,
                                                 cache=xy_provider_cache,
                                                 verbose=xy_provider_verbose),
                                     # Ensure that all batches have the same size
                                     (len(val_id_type_list) // batch_size) * batch_size,
                                     seed=seed,
                                     batch_size=batch_size),
        nb_val_samples=nb_val_samples,
        callbacks=callbacks,
        verbose=verbose,
    )

    # save the last
    val_loss = history.history['val_loss'][-1]
    weights_filename = weights_filename.format(epoch=nb_epochs, val_loss=val_loss)
    model.save_weights(weights_filename)

    return history


def segmentation_validate(model,
                          val_id_type_list,
                          batch_size=16,
                          xy_provider_cache=None,
                          image_size=(224, 224)):

    xy_provider = cached_image_mask_provider
    xy_provider_label_type = 'trainval_label_0'
    val_iter = ImageMaskIterator(xy_provider(val_id_type_list,
                                             mask_type=xy_provider_label_type,
                                             test_mode=True,
                                             cache=xy_provider_cache,
                                             image_size=image_size),
                                 len(val_id_type_list),
                                 None,  # image generator
                                 batch_size=batch_size,
                                 data_format='channels_first')

    mean_jaccard_index = 0.0
    total_counter = 0
    for x, y_true, info in val_iter:
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


# ##### Classification ####

def classification_train(model,
                         train_id_type_list,
                         val_id_type_list,
                         batch_size=16,
                         nb_epochs=10,
                         lrate_decay_f=None,
                         samples_per_epoch=2048,
                         nb_val_samples=1024,
                         xy_provider_cache=None,
                         seed=None,
                         save_prefix="",
                         verbose=1):

    samples_per_epoch = (samples_per_epoch // batch_size) * batch_size
    nb_val_samples = (nb_val_samples // batch_size) * batch_size

    if not os.path.exists('weights'):
        os.mkdir('weights')

    weights_filename = os.path.join("weights", save_prefix + "_{epoch:02d}-{val_loss:.4f}.h5")
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True)
    callbacks = [model_checkpoint, ]
    if lrate_decay_f is not None:
        lrate = LearningRateScheduler(lrate_decay_f)
        callbacks.append(lrate)

    print("-- Training parameters: ", batch_size, nb_epochs, samples_per_epoch, nb_val_samples)

    xy_provider = cached_image_label_provider
    xy_provider_verbose = 0

    train_gen = ImageDataGenerator(pipeline=('random_transform', random_rgb_to_green, 'standardize'),
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=90.,
                                   width_shift_range=0.15, height_shift_range=0.15,
                                   shear_range=3.14/6.0,
                                   zoom_range=0.25,
                                   channel_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='constant')
    val_gen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=90.,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='constant')

    print("-- Fit stats of train dataset")
    train_gen.fit(xy_provider(train_id_type_list,
                              test_mode=True,
                              cache=xy_provider_cache),
                  len(train_id_type_list),
                  augment=True,
                  seed=seed,
                  save_to_dir=GENERATED_DATA,
                  save_prefix=save_prefix,
                  batch_size=4,
                  verbose=verbose)

    val_gen.mean = train_gen.mean
    val_gen.std = train_gen.std
    val_gen.principal_components = train_gen.principal_components

    print("-- Fit model")
    history = model.fit_generator(
        train_gen.flow(xy_provider(train_id_type_list,
                                   cache=xy_provider_cache,
                                   verbose=xy_provider_verbose),
                       # Ensure that all batches have the same size
                       (len(train_id_type_list) // batch_size) * batch_size,
                       seed=seed,
                       batch_size=batch_size),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epochs,
        validation_data=val_gen.flow(xy_provider(val_id_type_list,
                                                 cache=xy_provider_cache,
                                                 verbose=xy_provider_verbose),
                                     # Ensure that all batches have the same size
                                     (len(val_id_type_list) // batch_size) * batch_size,
                                     seed=seed,
                                     batch_size=batch_size),
        nb_val_samples=nb_val_samples,
        callbacks=callbacks,
        verbose=verbose,
    )

    # save the last
    val_loss = history.history['val_loss'][-1]
    weights_filename = weights_filename.format(epoch=nb_epochs, val_loss=val_loss)
    model.save_weights(weights_filename)

    return history


def classification_validate(model,
                            val_id_type_list,
                            batch_size=16,
                            xy_provider_cache=None):

    xy_provider = cached_image_label_provider
    val_iter = ImageDataIterator(xy_provider(val_id_type_list,
                                             test_mode=True,
                                             cache=xy_provider_cache),
                                 len(val_id_type_list),
                                 None,  # image generator
                                 batch_size=batch_size,
                                 data_format='channels_first')

    total_loss = 0.0
    total_counter = 0
    for x, y_true, info in val_iter:
        s = y_true.shape[0]
        total_counter += s
        y_pred = model.predict(x)
        loss = logloss_mc(y_true, y_pred)
        total_loss += s * loss
        print("--", total_counter, "batch loss : ", loss, " | info:", info)

    if total_counter == 0:
        total_counter += 1

    total_loss *= 1.0 / total_counter
    print("Total loss : ", total_loss)


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

