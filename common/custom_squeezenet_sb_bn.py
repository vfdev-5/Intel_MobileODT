################################################################################
# Custom SqueezeNet with simple bypass and BatchNormalization
################################################################################

import os
import sys

from keras.layers import Concatenate, Add, Dropout, Convolution2D, MaxPooling2D, Input
from keras.layers import Activation, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD
from keras import backend as K
from keras_metrics import precision, recall

def get_cnn(optimizer='adam', lr=0.01, add_simple_bypass=False):
    """
        CNN joins two SqueezeNets at fire-5 module and the rest as a
        single squeezeent
        Same as get_cnn but with all fire modules as fire_module_bn
    """
    input_shape1 = (299, 299, 3)
    input_shape2 = (299, 299, 3)
    n_classes = 3

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    input1 = Input(shape=input_shape1)
    input2 = Input(shape=input_shape2)

    x1 = _get_5_firemodules(input1, channel_axis=channel_axis, name_suffix='_1', add_simple_bypass=add_simple_bypass)
    x2 = _get_5_firemodules(input2, channel_axis=channel_axis, name_suffix='_2', add_simple_bypass=add_simple_bypass)
    x = Concatenate(axis=channel_axis, name="double_fire5/concat")([x1, x2])

    # Repeat SqueezeNet from fire-7 module
    x = fire_module_bn(x, fire_id=6, squeeze=2 * 48, expand=192)
    # fire7 and bypass
    if add_simple_bypass:
        x1 = x
        x2 = fire_module_bn(x, fire_id=7, squeeze=2 * 48, expand=192)
        x = Add()([x1, x2])
    else:
        x = fire_module_bn(x, fire_id=7, squeeze=2 * 48, expand=192)
    
    x = fire_module_bn(x, fire_id=8, squeeze=2 * 64, expand=256)
    # fire9 and bypass
    if add_simple_bypass:    
        x1 = x
        x2 = fire_module_bn(x, fire_id=9, squeeze=2 * 64, expand=256)
        x = Add()([x1, x2])    
    else:
        x = fire_module_bn(x, fire_id=9, squeeze=2 * 64, expand=256)
        
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(n_classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='loss')(x)

    model = Model(inputs=[input1, input2], outputs=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', precision, recall])
    model.name = "Custom_double_SqueezeNet"
    return model


def _get_5_firemodules(input_layer, channel_axis, name_suffix='', add_simple_bypass=False):

    x = Convolution2D(64, (3, 3),
                      strides=(2, 2), padding='valid',
                      kernel_initializer='glorot_normal',
                      name='conv1'+name_suffix)(input_layer)
    x = BatchNormalization(axis=channel_axis, name='bn1'+name_suffix)(x)
    x = Activation('relu', name='relu_conv1'+name_suffix)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1'+name_suffix)(x)

    x = fire_module_bn(x, fire_id=2, squeeze=16, expand=64, name_suffix=name_suffix)
    # fire3 and bypass
    if add_simple_bypass:
        x1 = x
        x2 = fire_module_bn(x, fire_id=3, squeeze=16, expand=64, name_suffix=name_suffix)
        x = Add()([x1, x2])
    else:
        x = fire_module_bn(x, fire_id=3, squeeze=16, expand=64, name_suffix=name_suffix)
        
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3'+name_suffix)(x)

    x = fire_module_bn(x, fire_id=4, squeeze=32, expand=128, name_suffix=name_suffix)
    # fire5 and bypass
    if add_simple_bypass:
        x1 = x
        x2 = fire_module_bn(x, fire_id=5, squeeze=32, expand=128, name_suffix=name_suffix)
        x = Add()([x1, x2])
    else:
        x = fire_module_bn(x, fire_id=5, squeeze=32, expand=128, name_suffix=name_suffix)
        
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5'+name_suffix)(x)
    return x


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


def fire_module_bn(x, fire_id, squeeze=16, expand=64, name_suffix=''):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1),
                      padding='valid',
                      kernel_initializer='glorot_normal',
                      name=s_id + sq1x1 + name_suffix)(x)
    x = BatchNormalization(axis=channel_axis, name=s_id + sq1x1 + '_bn' + name_suffix)(x)
    x = Activation('relu', name=s_id + relu + sq1x1 + name_suffix)(x)

    left = Convolution2D(expand, (1, 1),
                         padding='valid',
                         kernel_initializer='glorot_normal',
                         name=s_id + exp1x1 + name_suffix)(x)
    left = BatchNormalization(axis=channel_axis, name=s_id + exp1x1 + '_bn' + name_suffix)(left)
    left = Activation('relu', name=s_id + relu + exp1x1 + name_suffix)(left)

    right = Convolution2D(expand, (3, 3),
                          padding='same',
                          kernel_initializer='glorot_normal',
                          name=s_id + exp3x3 + name_suffix)(x)
    right = BatchNormalization(axis=channel_axis, name=s_id + exp3x3 + '_bn' + name_suffix)(right)
    right = Activation('relu', name=s_id + relu + exp3x3 + name_suffix)(right)

    x = Concatenate(axis=channel_axis, name=s_id + 'concat' + name_suffix)([left, right])
    return x