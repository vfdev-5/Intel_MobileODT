from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta
from keras import __version__

assert __version__ == '1.2.2', "Wrong Keras version : %s" % __version__

set_image_dim_ordering('th')


def get_resnet_2(image_size=(224, 224)):
    base_model = ResNet50(include_top=False, input_tensor=None, input_shape=(3,) + image_size)
    for layer in base_model.layers:
        layer.trainable = False



def get_resnet50(image_size=(224, 224)):
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
    optimizer = Adadelta(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy',])
    return model
