from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import AveragePooling2D, Dropout, Dense, Flatten
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta, Adam
from keras import __version__

assert __version__ == '1.2.2', "Wrong Keras version : %s" % __version__

set_image_dim_ordering('th')


def get_vgg16(trained=True, finetuning=True, optimizer='adadelta', lr=0.01, image_size=(224, 224), names_to_train=None):
    """
    Method get VGG16 model from keras applications, adapt inputs and outputs and return compiled model
    """
    if not trained:
        assert not finetuning, "WTF"

    weights = 'imagenet' if trained else None
    base_model = VGG16(include_top=False, input_tensor=None, classes=3,
                              input_shape=(3,) + image_size[::-1], weights=weights)

    if finetuning:
        if names_to_train is None:
            names_to_train=[            
                #'block3_conv1',
                #'block3_conv2',
                #'block3_conv3',

                #'block4_conv1',
                #'block4_conv2',
                #'block4_conv3',

                'block5_conv1',
                'block5_conv2',
                'block5_conv3',
            ]
        for layer in base_model.layers:
            if layer.name in names_to_train:
                layer.trainable = True
            else:
                layer.trainable = False

    x = Flatten()(base_model.output)
    output = Dense(3, activation='softmax', name='output')(x)
    model = Model(input=base_model.input, output=output)


    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'precision', 'recall'])
    return model
