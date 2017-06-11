
import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
squeezenet_path = os.path.abspath(os.path.join(current_path, '..', 'common', 'KerasSqueezeNet'))
if squeezenet_path not in sys.path:
    sys.path.append(squeezenet_path)

from keras_squeezenet.squeezenet import SqueezeNet
from keras.optimizers import Adadelta, Adam, SGD


def get_squeezenet(optimizer='', lr=0.01, weights='imagenet'):
    """
    """
    input_shape = (299, 299, 3)

    names_to_train = [
        'fire6/squeeze1x1', 'fire6/expand1x1', 'fire6/expand3x3',
        'fire7/squeeze1x1', 'fire7/expand1x1', 'fire7/expand3x3',
        'fire8/squeeze1x1', 'fire8/expand1x1', 'fire8/expand3x3',
        'fire9/squeeze1x1', 'fire9/expand1x1', 'fire9/expand3x3',
        'conv10',
    ]

    n_classes = 3

    model = SqueezeNet(input_shape=input_shape, classes=n_classes, include_top=False, weights=weights)

    # Rename layers and set some of them trainable
    for layer in model.layers:
        if layer.name in names_to_train:
            layer.trainable = True
        else:
            layer.trainable = False

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        opt = None

    if opt is not None:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "SqueezeNet"
    return model

