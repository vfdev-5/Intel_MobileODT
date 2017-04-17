from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import AveragePooling2D, Dropout, Dense, Flatten
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta, Nadam
from keras import __version__

assert __version__ == '1.2.2', "Wrong Keras version : %s" % __version__

set_image_dim_ordering('th')


def get_inception(trained=True, finetuning=True):
    """
    Method get InceptionV3 model from keras applications, adapt inputs and outputs and return compiled model
    """
    if not trained:
        assert not finetuning, "WTF"

    weights = 'imagenet' if trained else None
    base_model = InceptionV3(include_top=False, input_tensor=None,
                              input_shape=(3, 299, 299), weights=weights)

    if finetuning:
        names_to_train=[
        ]
        for layer in base_model.layers:
            if layer.name in names_to_train:
                layer.trainable = True
            else:
                layer.trainable = False

    x = base_model.output
    x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.7)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(input=base_model.input, output=output)

    optimizer = Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', ])
    return model
