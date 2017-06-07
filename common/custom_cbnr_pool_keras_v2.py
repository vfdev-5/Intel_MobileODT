
import numpy as np

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Add, ZeroPadding2D, Flatten
from keras.layers import Convolution2D, Activation, MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam, Adadelta, SGD


def get_cnn(optimizer='adam', lr=0.01, n_filters_0=64):
    """

    """

    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3, 224, 224)
    else:
        channels_axis = -1
        input_shape = (224, 224, 3)
    n_classes = 3

    inputs = Input(input_shape)
    x = inputs
    x = block_cbnr(x, n_filters_0, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_1a')
    x = block_cbnr(x, n_filters_0, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_1b')
    x1a = x
    x1b = x

    x1b = block_cbnr(x1b, n_filters_0, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_1c')
    x1b = block_cbnr(x1b, n_filters_0 * 2, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_2a')
    x1b = block_cbnr(x1b, n_filters_0 * 2, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_2b')

    x2a = x1b
    x2b = x1b

    x1a = MaxPooling2D(pool_size=(2, 2), padding='same')(x1a)
    x2a = Concatenate(axis=channels_axis, name='cc_1a_2a')([x2a, x1a])
    x2a = block_cbnr(x2a, n_filters_0 * 2, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_12')
    x2a = GlobalMaxPooling2D()(x2a)

    x2b = block_cbnr(x2b, n_filters_0 * 2, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_2c')
    x2b = block_cbnr(x2b, n_filters_0 * 3, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_3a')

    x3a = x2b
    x3b = x2b

    x1a = MaxPooling2D(pool_size=(2, 2), padding='same')(x1a)
    x3a = Concatenate(axis=channels_axis, name='cc_1a_3a')([x3a, x1a])
    x3a = block_cbnr(x3a, n_filters_0 * 3, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_13')
    x3a = GlobalMaxPooling2D()(x3a)

    x3b = block_cbnr(x3b, n_filters_0 * 3, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_3b')
    x3b = block_cbnr(x3b, n_filters_0 * 4, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_4a')

    x4a = x3b
    x4b = x3b

    x1a = MaxPooling2D(pool_size=(2, 2), padding='same')(x1a)
    x4a = Concatenate(axis=channels_axis, name='cc_1a_4a')([x4a, x1a])
    x4a = block_cbnr(x4a, n_filters_0 * 4, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_14')
    x4a = GlobalMaxPooling2D()(x4a)

    x4b = block_cbnr(x4b, n_filters_0 * 4, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_4b')
    x4b = block_cbnr(x4b, n_filters_0 * 5, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_5a')
    x4b = block_cbnr(x4b, n_filters_0 * 5, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_5b')
    x4b = GlobalMaxPooling2D()(x4b)

    x1a = MaxPooling2D(pool_size=(2, 2), padding='same')(x1a)
    x1a = GlobalMaxPooling2D()(x1a)

    x = Concatenate(axis=channels_axis)([x1a, x2a, x3a, x4a, x4b])
    outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "cbnr_pool_1"

    return model


def get_cnn2(optimizer='adam', lr=0.01, n_filters_0=64, n_filters_1=512):

    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3, 224, 224)
    else:
        channels_axis = -1
        input_shape = (224, 224, 3)
    n_classes = 3

    inputs = Input(input_shape)
    x = inputs

    x = block_cbnr(x, n_filters_0, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_1a')
    x = block_cbnr(x, n_filters_0, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_1b')
    x = block_cbnr(x, n_filters_0, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_1c')

    x1a = x
    x1b = x
    x1a = GlobalMaxPooling2D()(x1a)

    x = block_cbnr(x1b, n_filters_0, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_1d')

    x = block_cbnr(x, n_filters_0 * 2, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_2a')
    x = block_cbnr(x, n_filters_0 * 2, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_2b')
    x = block_cbnr(x, n_filters_0 * 2, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_2c')

    x2a = x
    x2b = x
    x2a = GlobalMaxPooling2D()(x2a)

    x = block_cbnr(x2b, n_filters_0 * 2, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_2d')

    x = block_cbnr(x, n_filters_0 * 4, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_3a')
    x = block_cbnr(x, n_filters_0 * 4, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_3b')
    x = block_cbnr(x, n_filters_0 * 4, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_3c')

    x3a = x
    x3b = x
    x3a = GlobalMaxPooling2D()(x3a)

    x = block_cbnr(x3b, n_filters_0 * 4, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_3d')

    x = block_cbnr(x, n_filters_0 * 8, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_4a')
    x = block_cbnr(x, n_filters_0 * 8, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_4b')
    x = block_cbnr(x, n_filters_0 * 8, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_4c')

    x4a = x
    x4a = GlobalMaxPooling2D()(x4a)

    x = Concatenate(axis=channels_axis)([x1a, x2a, x3a, x4a])

    x = Dense(n_filters_1, name='fc1')(x)
    x = BatchNormalization(name='fc1_bn')(x)
    x = Activation('relu', name='fc1_act')(x)

    x = Dense(n_filters_1, name='fc2')(x)
    x = BatchNormalization(name='fc2_bn')(x)
    x = Activation('relu', name='fc2_act')(x)

    outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "cbnr_pool_1"

    return model


def get_cnn3(optimizer='adam', lr=0.01, n_filters_0=64, n_filters_1=512):

    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3, 224, 224)
    else:
        channels_axis = -1
        input_shape = (224, 224, 3)
    n_classes = 3

    inputs = Input(input_shape)
    x = inputs

    x = block_cbnr(x, n_filters_0, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_1a')

    x1a = x
    x1b = x
    x1a = GlobalMaxPooling2D()(x1a)

    x = block_cbnr(x1b, n_filters_0, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_1b')

    x = block_cbnr(x, n_filters_0 * 2, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_2a')

    x2a = x
    x2b = x
    x2a = GlobalMaxPooling2D()(x2a)

    x = block_cbnr(x2b, n_filters_0 * 2, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_2b')

    x = block_cbnr(x, n_filters_0 * 4, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_3a')
    x = block_cbnr(x, n_filters_0 * 4, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_3b')

    x3a = x
    x3b = x
    x3a = GlobalMaxPooling2D()(x3a)

    x = block_cbnr(x3b, n_filters_0 * 4, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_3c')

    x = block_cbnr(x, n_filters_0 * 8, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_4a')
    x = block_cbnr(x, n_filters_0 * 8, (3, 3), (1, 1), channels_axis=channels_axis, s_id='_4b')
    x = block_cbnr(x, n_filters_0 * 8, (3, 3), (2, 2), channels_axis=channels_axis, s_id='_4c')

    x4a = x
    x4a = GlobalMaxPooling2D()(x4a)

    x = Concatenate(axis=channels_axis)([x1a, x2a, x3a, x4a])

    x = Dense(n_filters_1, name='fc1')(x)
    x = BatchNormalization(name='fc1_bn')(x)
    x = Activation('relu', name='fc1_act')(x)

    x = Dense(n_filters_1, name='fc2')(x)
    x = BatchNormalization(name='fc2_bn')(x)
    x = Activation('relu', name='fc2_act')(x)

    outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "cbnr_pool_1"

    return model


def get_conv_dense(optimizer='adam', lr=0.01, n_filters_0=64):
    """

    """

    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3, 224, 224)
    else:
        channels_axis = -1
        input_shape = (224, 224, 3)
    n_classes = 3

    inputs = Input(input_shape)
    x = inputs
    x = ZeroPadding2D((3, 3))(x)
    x = Convolution2D(n_filters_0, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=channels_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = Flatten()(x)

    outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "cbnr_pool_1"

    return model


def block_cbnr(input_layer, n_filters, kernel_size=(3, 3), 
               strides=(1, 1), channels_axis=1, s_id=''):
    x = Convolution2D(n_filters, kernel_size,
                      strides=strides,
                      padding='same',
                      use_bias=False, name='cbnr' + s_id + '_conv')(input_layer)
    x = BatchNormalization(axis=channels_axis, name='cbnr' + s_id + '_bn')(x)
    x = Activation('relu', name='cbnr' + s_id + '_act')(x)
    return x

