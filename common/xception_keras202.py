from keras.applications.xception import Xception
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.backend import set_image_data_format
from keras.optimizers import Adadelta, Nadam, Adam, RMSprop
from keras import __version__

assert __version__ == '2.0.2', "Wrong Keras version : %s" % __version__

set_image_data_format('channels_last')


def get_xception(trained=True, finetuning=True, optimizer='adadelta', lr=0.01):
    """
    Method get Xception model from keras applications, adapt inputs and outputs and return compiled model
    """
    if not trained:
        assert not finetuning, "WTF"

    weights = 'imagenet' if trained else None
    base_model = Xception(include_top=False, input_tensor=None,
                          input_shape=(299, 299, 3), weights=weights,
                          pooling='avg')

    if finetuning:
        names_to_train=[
            # 'block13_sepconv1', 'block13_sepconv1_bn',
            # 'block13_sepconv2', 'block13_sepconv2_bn',
            # 'block14_sepconv1', 'block14_sepconv1_bn',
            'block14_sepconv2', 'block14_sepconv2_bn',
        ]
        for layer in base_model.layers:
            if layer.name in names_to_train:
                layer.trainable = True
            else:
                layer.trainable = False

    x = base_model.output
    x = Dropout(0.5)(x)
    # output = Dense(3, activation='softmax')(x)
    output = Dense(3, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    
    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=lr, decay=0.01)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', ])        
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', ])
    return model


def get_xception_3_dense(finetuning=True):
    """
    Method get Xception model from keras applications, adapt inputs and outputs and return compiled model
    """
    base_model = Xception(include_top=False, input_tensor=None,
                          input_shape=(299, 299, 3), weights='imagenet',
                          pooling='avg')

    if finetuning:
        names_to_train=[
            # 'block13_sepconv1', 'block13_sepconv1_bn',
            # 'block13_sepconv2', 'block13_sepconv2_bn',
            # 'block14_sepconv1', 'block14_sepconv1_bn',
            'block14_sepconv2', 'block14_sepconv2_bn',
        ]
        for layer in base_model.layers:
            if layer.name in names_to_train:
                layer.trainable = True
            else:
                layer.trainable = False

    x = base_model.output
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.7)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    optimizer = Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', ])
    return model
