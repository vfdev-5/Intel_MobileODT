from keras import __version__ as keras_version
assert keras_version == '1.2.2', "Wrong Keras version : %s" % keras_version

from keras.models import Model
from keras.layers import Convolution2D, Input, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Dropout, Dense, Flatten
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta, Adam, Nadam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.regularizers import l2

set_image_dim_ordering('th')


def base_conv(x, n_filters, subsample=(2, 2), activation='relu', regularization=False):
    regularizer = l2 if regularization else None
    x = Convolution2D(n_filters, 3, 3, 
                      subsample=subsample, 
                      border_mode='valid', W_regularizer=regularizer)(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    return Activation(activation)(x)


def base_conv2(x, n_filters, subsample=(2, 2), activation='relu', regularization=False):
    regularizer = l2 if regularization else None
    x = Convolution2D(n_filters, 3, 3, border_mode='valid', W_regularizer=regularizer)(x)
    x = Convolution2D(n_filters, 3, 3, subsample=subsample, border_mode='valid', W_regularizer=regularizer)(x)    
    x = BatchNormalization(mode=2, axis=1)(x)
    return Activation(activation)(x)


def get_cnn_1(n_filters=32, optimizer='adam', lr=0.01):
        
    input = Input((3, 299, 299))
    x = input   
    x = base_conv(x, n_filters)
    x = base_conv(x, n_filters)

    x = base_conv(x, 2 * n_filters)
    x = base_conv(x, 2 * n_filters)

    x = base_conv(x, 4 * n_filters)
    x = base_conv(x, 4 * n_filters)
    
    x = Flatten()(x)
    x = Dense(4 * n_filters, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2 * n_filters, activation='relu')(x)    
    x = Dropout(0.5)(x)
    x = Dense(n_filters, activation='relu')(x)    
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model


def get_cnn_2(n_filters=16, optimizer='adam', lr=0.01):
        
    input = Input((3, 299, 299))
    x = input   
    x = base_conv(x, n_filters)
    x = base_conv(x, n_filters)

    x = base_conv(x, 2 * n_filters)
    x = base_conv(x, 2 * n_filters)

    x = base_conv(x, 4 * n_filters)
    x = base_conv(x, 4 * n_filters)
    
    x = Flatten()(x)
    x = Dense(n_filters, activation='relu')(x)    
    x = Dropout(0.7)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model

def get_cnn_3(n_filters=16, optimizer='adam', lr=0.01):
        
    input = Input((3, 299, 299))
    x = input   
    x = base_conv(x, n_filters, subsample=(1, 1))
    x = base_conv(x, n_filters)
    x = base_conv(x, n_filters)

    x = base_conv(x, 2 * n_filters)
    x = base_conv(x, 2 * n_filters)

    x = base_conv(x, 4 * n_filters)
    x = base_conv(x, 4 * n_filters)
    
    x = Flatten()(x)
    x = Dense(2 * n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)
    x = Dense(n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model

def get_cnn_4(n_filters=32, optimizer='adam', lr=0.01):
        
    input = Input((3, 299, 299))
    x = input   
    x = base_conv(x, n_filters, subsample=(1, 1))
    x = base_conv(x, n_filters)
    x = base_conv(x, n_filters)

    x = base_conv(x, 2 * n_filters, subsample=(1, 1))
    x = base_conv(x, 2 * n_filters)
    x = base_conv(x, 2 * n_filters)

    x = base_conv(x, 4 * n_filters)
    x = base_conv(x, 4 * n_filters)
    
    x = Flatten()(x)
    x = Dense(n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)
    output = Dense(3, activation='softmax')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model

def get_cnn_5(n_filters=64, optimizer='adam', lr=0.01):
        
    input = Input((3, 64, 64))
    x = input   
    x = base_conv2(x, n_filters, activation='elu')
    x = base_conv2(x, 2 * n_filters, activation='elu')
    x = base_conv2(x, 4 * n_filters, activation='elu')
    x = base_conv2(x, 8 * n_filters, activation='elu')
    
    x = Flatten()(x)
    x = Dense(4 * n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)

    x = Dense(2 * n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)

    x = Dense(n_filters, activation='elu')(x)    
    x = Dropout(0.65)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input=input, output=output)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'recall', 'precision'])
    return model


def get_alexnet(optimizer='adam', lr=0.01):
    
    input = Input((3, 64, 64))
    x = input   
    
    x = Convolution2D(96, 11, 11, subsample=(2,2), border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid')(x)
    
    x = Convolution2D(384, 5, 5, subsample=(1,1), border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid')(x)
    
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(384, 3, 3, border_mode='same', activation='relu')(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(384, 3, 3, border_mode='same', activation='relu')(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2,2), border_mode='valid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(3)(x)
    output = Activation('softmax')(x)
    
    model = Model(input=input, output=output)
    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'precision', 'recall'])

    return model

