from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta, Nadam
from keras import __version__

assert __version__ == '1.2.2', "Wrong Keras version : %s" % __version__

set_image_dim_ordering('th')


def get_resnet50(image_size=(224, 224), opt='nadam'):
    """
    Method get ResNet50 model from keras applications, adapt inputs and outputs and return compiled model
    """
    base_model = ResNet50(include_top=False, input_tensor=None, input_shape=(3,) + image_size)
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(input=base_model.input, output=output)

    if opt == 'nadam':
        optimizer = Nadam()
    elif opt == 'adadelta':
        optimizer = Adadelta()
    else:
        raise Exception("optimizer is unknown")

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', ])
    return model


def get_resnet_original(image_size=(224, 224), opt='nadam', trained=False):
    """
    Method get not trained ResNet50 original model from keras applications,
    adapt inputs and outputs and return compiled model
    """
    weights = 'imagenet' if trained else None
    base_model = ResNet50(include_top=False, weights=weights, input_tensor=None, input_shape=(3,) + image_size)
    x = Flatten()(base_model.output)
    output = Dense(3, activation='softmax')(x)
    model = Model(input=base_model.input, output=output)
    if opt == 'nadam':
        optimizer = Nadam()
    elif opt == 'adadelta':
        optimizer = Adadelta()
    else:
        raise Exception("optimizer is unknown")
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', ])
    return model
