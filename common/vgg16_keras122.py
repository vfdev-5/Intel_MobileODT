from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import AveragePooling2D, Dropout, Dense, Flatten
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta, Adam
from keras import __version__

assert __version__ == '1.2.2', "Wrong Keras version : %s" % __version__

set_image_dim_ordering('th')


def get_vgg16(trained=True, finetuning=True, optimizer='adadelta', lr=0.01):
    """
    Method get VGG16 model from keras applications, adapt inputs and outputs and return compiled model
    """
    if not trained:
        assert not finetuning, "WTF"

    weights = 'imagenet' if trained else None
    base_model = VGG16(include_top=True, input_tensor=None, classes=3,
                              input_shape=(3, 224, 224), weights=weights)

    if finetuning:
        names_to_train=[]
        for layer in base_model.layers:
            if layer.name in names_to_train:
                layer.trainable = True
            else:
                layer.trainable = False

    model = base_model

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', ])
    return model
