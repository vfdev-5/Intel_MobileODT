
import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
squeezenet_path = os.path.abspath(os.path.join(current_path, '..', 'common', 'KerasSqueezeNet'))
if squeezenet_path not in sys.path:
    sys.path.append(squeezenet_path)

from keras_squeezenet.squeezenet import SqueezeNet, fire_module
from keras.layers import concatenate, Dropout, Convolution2D, Activation, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD
from keras import backend as K
from keras_metrics import precision, recall


def get_cnn(optimizer='adam', lr=0.01):
    """
        CNN joins two SqueezeNets at fire-6 module
    """
    input_shape1 = (299, 299, 3)
    input_shape2 = (299, 299, 3)

    names_to_train = [
        'fire5/squeeze1x1', 'fire5/expand1x1', 'fire5/expand3x3',
        'fire6/squeeze1x1', 'fire6/expand1x1', 'fire6/expand3x3',
        'fire7/squeeze1x1', 'fire7/expand1x1', 'fire7/expand3x3',
        'fire8/squeeze1x1', 'fire8/expand1x1', 'fire8/expand3x3',
        'fire9/squeeze1x1', 'fire9/expand1x1', 'fire9/expand3x3',
        'conv10',
    ]

    n_classes = 3

    snet1 = SqueezeNet(input_shape=input_shape1, classes=n_classes, include_top=False)
    snet2 = SqueezeNet(input_shape=input_shape2, classes=n_classes, include_top=False)

    for i, cnn in enumerate([snet1, snet2]):
        # Rename layers and set some of them trainable
        for layer in cnn.layers:
            if layer.name in names_to_train:
                layer.trainable = True
            else:
                layer.trainable = False
            layer.name += '_%i' % (i+1)

    snet1.name = "SqueezeNet_1"
    snet2.name = "SqueezeNet_2"

    x1 = snet1.get_layer('fire6/concat_1').output
    x2 = snet2.get_layer('fire6/concat_2').output

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = concatenate([x1, x2], axis=channel_axis, name="double_fire6/concat")

    # Repeat SqueezeNet from fire-7 module
    x = fire_module(x, fire_id=7, squeeze=2*48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=2*64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=2*64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    name = 'conv10'
    x = Convolution2D(n_classes, (1, 1), padding='valid', name=name)(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(inputs=[snet1.inputs[0], snet2.inputs[0]], outputs=out)

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