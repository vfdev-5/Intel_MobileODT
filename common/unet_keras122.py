
from keras.layers import Dense, Flatten, Input, Convolution2D, Activation, MaxPooling2D, UpSampling2D, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras.backend import set_image_dim_ordering

set_image_dim_ordering('th')


def base_conv(x, n_filters):
    x = Convolution2D(n_filters, 3, 3, border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=1)(x)
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
        x = merge([x, list_encoder[l-i-2]], mode='concat', concat_axis=1)
        x = base_conv(x, list_nb_filters[l-i-2])
        x = base_conv(x, list_nb_filters[l-i-2])
        temp_layers.append(x)

    return temp_layers[-1]


def get_unet(image_size=(224, 224), n_filters=32):

    inputs = Input((3,) + image_size)

    list_encoder, list_nb_filters = encoder(inputs, n_filters)
    x = decoder(list_encoder, list_nb_filters)
    x = Flatten()(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(input=inputs, output=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy',])
    return model
