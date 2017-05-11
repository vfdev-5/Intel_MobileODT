
import os
import numpy as np

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras import __version__ as keras_version

# Project
from data_utils import GENERATED_DATA
from metrics import logloss_mc
from xy_providers import cached_image_label_provider


# Local keras-contrib:
from preprocessing.image.generators import ImageDataGenerator


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
                                    "val_acc={val_acc:.4f}_val_precision={val_precision:.4f}_" +
                                    "val_recall={val_recall:.4f}.h5")

    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, min_delta=0.01, verbose=0)
    
    callbacks = [model_checkpoint, early_stop, ]
    if lrate_decay_f is not None:
        lrate = LearningRateScheduler(lrate_decay_f)
        callbacks.append(lrate)

    print("\n-- Training parameters: %i, %i, %i, %i" % (batch_size, nb_epochs, samples_per_epoch, nb_val_samples))

    xy_provider = cached_image_label_provider
    xy_provider_verbose = 0

    train_gen = ImageDataGenerator(pipeline=('random_transform', 'standardize'),
                                   featurewise_center=normalize_data,
                                   featurewise_std_normalization=normalize_data,
                                   rotation_range=90.,
                                   # width_shift_range=0.25, height_shift_range=0.25,
                                   # shear_range=3.14/6.0,
                                   zoom_range=[0.65, 1.2],
                                   # channel_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='constant')
    val_gen = ImageDataGenerator(rotation_range=15.,
                                 featurewise_center=normalize_data,
                                 featurewise_std_normalization=normalize_data,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='constant')

    if hasattr(K, 'image_data_format'):
        channels_first = K.image_data_format() == 'channels_first'
    elif hasattr(K, 'image_dim_ordering'):
        channels_first = K.image_dim_ordering() == 'th'
    else:
        raise Exception("Failed to find backend data format")

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
                          save_prefix=save_prefix,
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
            m = np.array([123.68, 116.779, 103.939]) / 255.0 # RGB
            if channels_first:                
                m = m[:, None, None]
            else:
                m = m[None, None, :]
            train_gen.mean = m
                                          
    val_gen.mean = train_gen.mean
    val_gen.std = train_gen.std
    val_gen.principal_components = train_gen.principal_components

    print("\n-- Fit model")
    try:

        train_flow = train_gen.flow(xy_provider(train_id_type_list,
                                                image_size=image_size,
                                                option=option,
                                                channels_first=channels_first,
                                                cache=xy_provider_cache,
                                                verbose=xy_provider_verbose),
                                    # Ensure that all batches have the same size
                                    (len(train_id_type_list) // batch_size) * batch_size,
                                    seed=seed,
                                    batch_size=batch_size)

        val_flow = val_gen.flow(xy_provider(val_id_type_list,
                                            image_size=image_size,
                                            option=option,
                                            channels_first=channels_first,
                                            cache=xy_provider_cache,
                                            verbose=xy_provider_verbose),
                                # Ensure that all batches have the same size
                                (len(val_id_type_list) // batch_size) * batch_size,
                                seed=seed,
                                batch_size=batch_size)

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
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_acc'][-1]
        val_recall = history.history['val_recall'][-1]
        val_precision = history.history['val_precision'][-1]
        weights_filename = weights_filename.format(epoch=nb_epochs,
                                                   val_loss=val_loss,
                                                   val_acc=val_acc,
                                                   val_precision=val_precision,
                                                   val_recall=val_recall)
        model.save_weights(weights_filename)
        return history

    except KeyboardInterrupt:
        return None


def classification_validate(model,
                            val_id_type_list,
                            option=None,
                            normalize_data=True,
                            normalization='',
                            image_size=(224, 224),
                            save_prefix="",
                            batch_size=16,
                            xy_provider_cache=None):

    
    if hasattr(K, 'image_data_format'):
        channels_first = K.image_data_format() == 'channels_first'
    elif hasattr(K, 'image_dim_ordering'):
        channels_first = K.image_dim_ordering() == 'th'
    else:
        raise Exception("Failed to find backend data format")

    xy_provider = cached_image_label_provider

    val_gen = ImageDataGenerator(featurewise_center=normalize_data,
                                 featurewise_std_normalization=normalize_data)

    if normalize_data:
        if normalization == '':
            assert len(save_prefix) > 0, "WTF"
            # Load mean, std, principal_components if file exists
            filename = os.path.join(GENERATED_DATA, save_prefix + "_stats.npz")
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
            
    flow = val_gen.flow(xy_provider(val_id_type_list,
                                    image_size=image_size,
                                    option=option,
                                    channels_first=channels_first,
                                    cache=xy_provider_cache,
                                    test_mode=True),
                        # Ensure that all batches have the same size
                        len(val_id_type_list),
                        batch_size=batch_size)

    total_loss = 0.0
    total_counter = 0
    for x, y_true, info in flow:
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