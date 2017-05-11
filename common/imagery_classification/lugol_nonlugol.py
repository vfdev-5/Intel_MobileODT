
import os

import numpy as np
import cv2

from keras import __version__ as keras_version
assert keras_version == '1.2.2', "Wrong Keras version : %s" % keras_version

from keras.models import Model
from keras.layers import Convolution2D, Input, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Dropout, Dense, Flatten
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta, Adam, Nadam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.regularizers import l2

set_image_dim_ordering('th')

from image_utils import get_cervix_image
from data_utils import GENERATED_DATA

# Local keras-contrib:
from preprocessing.image.generators import ImageDataGenerator


# ###### XY_provider ######

def cached_image_provider(image_id_type_list,
                          image_size=(224, 224),
                          channels_first=True,
                          cache=None,
                          verbose=0):
    if cache is None:
        cache = DataCache(n_samples=500)

    for i, (image_id, image_type) in enumerate(image_id_type_list):
        if verbose > 0:
            print("Image id/type:", image_id, image_type, "| counter=", i)

        key = (image_id, image_type)
        if key in cache:
            img, _ = cache.get(key)
            if channels_first:
                if img.shape[1:] != image_size[::-1]:
                    img = img.transpose([1, 2, 0])
                    img = cv2.resize(img, dsize=image_size[::-1])
                    img = img.transpose([2, 0, 1])
            else:
                if img.shape[:2] != image_size[::-1]:
                    img = cv2.resize(img, dsize=image_size[::-1])

        else:
            img = get_cervix_image(image_id, image_type)
            if img.dtype.kind is not 'u':
                if verbose > 0:
                    print("Image is corrupted. Id/Type:", image_id, image_type)
                continue
            img = cv2.resize(img, dsize=image_size[::-1])
            if channels_first:
                img = img.transpose([2, 0, 1])
            img = img.astype(np.float32) / 255.0
            cache.put(key, (img, None))
            
        yield img, None, (image_id, image_type)



def cached_image_label_provider(image_id_type_list,
                                image_size,
                                cache,
                                channels_first=True,
                                test_mode=False,
                                verbose=0):

    counter = 0
    image_id_type_list = list(image_id_type_list)
    while True:
        np.random.shuffle(image_id_type_list)
        for i, ((image_id, image_type), target) in enumerate(image_id_type_list):
            target = np.array([target, ])

            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", i)

            key = (image_id, image_type)
            if key in cache:
                if verbose > 0:
                    print("-- Load from RAM")
                img, label = cache.get(key)

            else:
                if verbose > 0:
                    print("-- Load from disk")

                img = get_cervix_image(image_id, image_type)

                if img.shape[:2] != image_size:
                    img = cv2.resize(img, dsize=image_size)

                if channels_first:
                    img = img.transpose([2, 0, 1])
                img = img.astype(np.float32) / 255.0
                # fill the cache only at first time:
                if counter == 0:
                    cache.put(key, (img, target))

            if test_mode:
                yield img, target, (image_id, image_type)
            else:
                yield img, target

        if test_mode:
            return
        counter += 1


# ###### Train model ######

def train(model,
          train_id_type_list,
          val_id_type_list,
          normalization='',
          batch_size=16,
          nb_epochs=10,
          class_weight={},
          lrate_decay_f=None,
          samples_per_epoch=2048,
          nb_val_samples=1024,
          xy_provider_cache=None,
          seed=None,
          save_prefix="",
          verbose=1):
    samples_per_epoch = (samples_per_epoch // batch_size + 1) * batch_size
    nb_val_samples = (nb_val_samples // batch_size + 1) * batch_size if nb_val_samples is not None else 0

    normalize_data = True
    image_size = (299, 299)

    if not os.path.exists('weights'):
        os.mkdir('weights')

    weights_filename = os.path.join("weights", save_prefix +
                                    "_{epoch:02d}_val_loss={val_loss:.4f}_" +
                                    "val_acc={val_acc:.4f}_val_precision={val_precision:.4f}_" +
                                    "val_recall={val_recall:.4f}.h5")
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True)
    callbacks = [model_checkpoint, ]
    if lrate_decay_f is not None:
        lrate = LearningRateScheduler(lrate_decay_f)
        callbacks.append(lrate)

    print("\n-- Training parameters: %i, %i, %i, %i" % (batch_size, nb_epochs, samples_per_epoch, nb_val_samples))

    xy_provider = cached_image_label_provider
    xy_provider_verbose = 0

    train_gen = ImageDataGenerator(pipeline=('random_transform', 'standardize'),
                                   featurewise_center=normalize_data,
                                   featurewise_std_normalization=normalize_data,
                                   rotation_range=45.,
                                   width_shift_range=0.05, height_shift_range=0.05,
                                   zoom_range=0.15,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='constant')
    if val_id_type_list is not None:
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
        elif normalization == 'resnet':
            print("Image normalization: ", normalization)
            train_gen.std = 1.0 / 255.0  # Rescale to [0.0, 255.0]
            m = np.array([123.68, 116.779, 103.939]) / 255.0  # RGB
            if channels_first:
                m = m[:, None, None]
            else:
                m = m[None, None, :]
            train_gen.mean = m

    if val_id_type_list is not None:
        val_gen.mean = train_gen.mean
        val_gen.std = train_gen.std
        val_gen.principal_components = train_gen.principal_components

    print("\n-- Fit model")
    try:

        train_flow = train_gen.flow(xy_provider(train_id_type_list,
                                                image_size=image_size,
                                                channels_first=channels_first,
                                                cache=xy_provider_cache,
                                                verbose=xy_provider_verbose),
                                    # Ensure that all batches have the same size
                                    (len(train_id_type_list) // batch_size) * batch_size,
                                    seed=seed,
                                    batch_size=batch_size)
        if val_id_type_list is not None:
            val_flow = val_gen.flow(xy_provider(val_id_type_list,
                                                image_size=image_size,
                                                channels_first=channels_first,
                                                cache=xy_provider_cache,
                                                verbose=xy_provider_verbose),
                                    # Ensure that all batches have the same size
                                    (len(val_id_type_list) // batch_size) * batch_size,
                                    seed=seed,
                                    batch_size=batch_size)
        else:
            val_flow = None
            nb_val_samples = None

        # New or old Keras API
        if int(keras_version[0]) == 2:
            print("- New Keras API found -")
            validation_steps = (nb_val_samples // batch_size) if val_flow is not None else None
            history = model.fit_generator(generator=train_flow,
                                          steps_per_epoch=(samples_per_epoch // batch_size),
                                          epochs=nb_epochs,
                                          validation_data=val_flow,
                                          validation_steps=validation_steps,
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
        pass
    
# ####### CNNs



def base_conv(x, n_filters, subsample=(2, 2), activation='relu', regularization=False):
    regularizer = l2 if regularization else None
    x = Convolution2D(n_filters, 3, 3, 
                      subsample=subsample, 
                      border_mode='valid', W_regularizer=regularizer)(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    return Activation(activation)(x)

def base_conv2(x, n_filters, subsample=(2, 2), activation='relu', regularization=False):
    regularizer = l2 if regularization else None
    x = Convolution2D(n_filters, 3, 3, border_mode='valid', W_regularizer=regularizer)(x)
    x = Convolution2D(n_filters, 3, 3, subsample=subsample, border_mode='valid', W_regularizer=regularizer)(x)    
    x = BatchNormalization(mode=2, axis=1)(x)
    return Activation(activation)(x)


def get_cnn_1(n_filters=32, optimizer='adam', lr=0.01):
        
    input = Input((3, 299, 299))
    x = input   
    x = base_conv(x, n_filters)
    x = base_conv(x, n_filters)

    x = base_conv(x, 2 * n_filters)
    x = base_conv(x, 2 * n_filters)

    x = base_conv(x, 4 * n_filters)
    x = base_conv(x, 4 * n_filters)
    
    x = Flatten()(x)
    x = Dense(4 * n_filters, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2 * n_filters, activation='relu')(x)    
    x = Dropout(0.5)(x)
    x = Dense(n_filters, activation='relu')(x)    
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model


def get_cnn_2(n_filters=16, optimizer='adam', lr=0.01):
        
    input = Input((3, 299, 299))
    x = input   
    x = base_conv(x, n_filters)
    x = base_conv(x, n_filters)

    x = base_conv(x, 2 * n_filters)
    x = base_conv(x, 2 * n_filters)

    x = base_conv(x, 4 * n_filters)
    x = base_conv(x, 4 * n_filters)
    
    x = Flatten()(x)
    x = Dense(n_filters, activation='relu')(x)    
    x = Dropout(0.7)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model

def get_cnn_3(n_filters=16, optimizer='adam', lr=0.01):
        
    input = Input((3, 299, 299))
    x = input   
    x = base_conv(x, n_filters, subsample=(1, 1))
    x = base_conv(x, n_filters)
    x = base_conv(x, n_filters)

    x = base_conv(x, 2 * n_filters)
    x = base_conv(x, 2 * n_filters)

    x = base_conv(x, 4 * n_filters)
    x = base_conv(x, 4 * n_filters)
    
    x = Flatten()(x)
    x = Dense(2 * n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)
    x = Dense(n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model

def get_cnn_4(n_filters=32, optimizer='adam', lr=0.01):
        
    input = Input((3, 299, 299))
    x = input   
    x = base_conv(x, n_filters, subsample=(1, 1))
    x = base_conv(x, n_filters)
    x = base_conv(x, n_filters)

    x = base_conv(x, 2 * n_filters, subsample=(1, 1))
    x = base_conv(x, 2 * n_filters)
    x = base_conv(x, 2 * n_filters)

    x = base_conv(x, 4 * n_filters)
    x = base_conv(x, 4 * n_filters)
    
    x = Flatten()(x)
    x = Dense(n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model

def get_cnn_5(n_filters=64, optimizer='adam', lr=0.01):
        
    input = Input((3, 64, 64))
    x = input   
    x = base_conv2(x, n_filters, activation='elu')
    x = base_conv2(x, 2 * n_filters, activation='elu')
    x = base_conv2(x, 4 * n_filters, activation='elu')
    x = base_conv2(x, 8 * n_filters, activation='elu')
    
    x = Flatten()(x)
    x = Dense(4 * n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)

    x = Dense(2 * n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)

    x = Dense(n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model


def get_alexnet(optimizer='adam', lr=0.01):
    
    input = Input((3, 64, 64))
    x = input   
    
    x = Convolution2D(96, 11, 11, subsample=(2,2), border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid')(x)
    
    x = Convolution2D(384, 5, 5, subsample=(1,1), border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid')(x)
    
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(384, 3, 3, border_mode='same', activation='relu')(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(384, 3, 3, border_mode='same', activation='relu')(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), border_mode='valid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1)(x)
    output = Activation('sigmoid')(x)
    
    model = Model(input=input, output=output)
    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'precision', 'recall'])

    return model
