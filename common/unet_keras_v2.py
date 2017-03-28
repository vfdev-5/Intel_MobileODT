
from keras.layers import Dense, Flatten, Input, Conv2D, Activation, MaxPooling2D, UpSampling2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.backend import set_image_dim_ordering
from keras import __version__

assert __version__ == '2.0.1', "Wring Keras version : %s" % __version__

set_image_dim_ordering('th')


def base_conv(x, n_filters):
    x = Conv2D(n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    return Activation('relu')(x)


def encoder(inputs, n_filters):

    list_encoder = []
    list_nb_filters = []
    temp_layers = [inputs]
    for i in range(5):
        nf = n_filters * 2**i
        x = base_conv(temp_layers[-1], nf)
        x = base_conv(x, nf)
        list_encoder.append(x)
        list_nb_filters.append(nf)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        temp_layers.append(x)

    i = 5
    nf = n_filters * 2**i
    x = base_conv(temp_layers[-1], nf)
    x = base_conv(x, nf)
    list_encoder.append(x)

    return list_encoder, list_nb_filters


def decoder(list_encoder, list_nb_filters):
    l = len(list_encoder)
    temp_layers = [list_encoder[l-1]]
    for i in range(l-1):
        x = UpSampling2D(size=(2, 2))(temp_layers[-1])
        x = Concatenate(axis=1)([x, list_encoder[l-i-2]])
        x = base_conv(x, list_nb_filters[l-i-2])
        x = base_conv(x, list_nb_filters[l-i-2])
        temp_layers.append(x)

    return temp_layers[-1]


def get_unet(image_size=(224, 224), n_filters=32):

    inputs = Input((3,) + image_size)

    list_encoder, list_nb_filters = encoder(inputs, n_filters)
    x = decoder(list_encoder, list_nb_filters)
    x = Flatten()(x)
    # x = Dense(100, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', ])
    return model
