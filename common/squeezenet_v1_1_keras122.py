from keras import __version__
assert __version__ == '1.2.2', "Wrong Keras version : %s" % __version__

from keras.models import Model
from keras.layers import MaxPooling2D, Dropout, Input, Convolution2D, ZeroPadding2D, \
    merge, Activation, GlobalAveragePooling2D
from keras import initializations
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta, Adam, SGD

set_image_dim_ordering('th')


def get_fire_squeeze1x1(input_layer, fire_index, n_filters):
    return Convolution2D(n_filters, 1, 1, 
                         border_mode='valid', 
                         init='glorot_normal',
                         activation='relu', 
                         name='fire%i/squeeze1x1' % fire_index)(input_layer)


def get_fire_expansion(input_layer, fire_index, n_filters):
    x1 = Convolution2D(n_filters, 1, 1, 
                       border_mode='valid', 
                       init='glorot_normal',
                       activation='relu', 
                       name='fire%i/expand1x1' % fire_index)(input_layer)
    
    x2 = ZeroPadding2D(padding=(1, 1))(input_layer)
    x2 = Convolution2D(n_filters, 3, 3, 
                       border_mode='valid', 
                       init='glorot_normal',
                       activation='relu', 
                       name='fire%i/expand3x3' % fire_index)(x2)
    return merge([x1, x2], mode='concat', concat_axis=1)


def get_base_squeezenet_v1_1(input_layer, n_classes):
    
    x = Convolution2D(64, 3, 3, 
                      subsample=(2, 2), border_mode='valid', 
                      init='glorot_normal',
                      activation='relu', name='conv1')(input_layer)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                     border_mode='valid', name='pool1')(x)
    
    x = get_fire_squeeze1x1(x, fire_index=2, n_filters=16)    
    x = get_fire_expansion(x, fire_index=2, n_filters=64)

    x = get_fire_squeeze1x1(x, fire_index=3, n_filters=16)    
    x = get_fire_expansion(x, fire_index=3, n_filters=64)
    
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                     border_mode='valid', name='pool3')(x)

    x = get_fire_squeeze1x1(x, fire_index=4, n_filters=32)    
    x = get_fire_expansion(x, fire_index=4, n_filters=128)

    x = get_fire_squeeze1x1(x, fire_index=5, n_filters=32)    
    x = get_fire_expansion(x, fire_index=5, n_filters=128)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                     border_mode='valid', name='pool5')(x)
    
    x = get_fire_squeeze1x1(x, fire_index=6, n_filters=48)    
    x = get_fire_expansion(x, fire_index=6, n_filters=192)

    x = get_fire_squeeze1x1(x, fire_index=7, n_filters=48)    
    x = get_fire_expansion(x, fire_index=7, n_filters=192)

    x = get_fire_squeeze1x1(x, fire_index=8, n_filters=64)    
    x = get_fire_expansion(x, fire_index=8, n_filters=256)
    
    x = get_fire_squeeze1x1(x, fire_index=9, n_filters=64)    
    x = get_fire_expansion(x, fire_index=9, n_filters=256)
    
    x = Dropout(0.5, name='drop9')(x)
    x = Convolution2D(n_classes, 1, 1, 
                      border_mode='valid', 
                      init=initializations.get('normal', scale=0.01),
                      activation='relu', name='conv10')(x)
    
    x = GlobalAveragePooling2D(name='pool10')(x)
    x = Activation('softmax')(x)
    model = Model(input=input_layer, output=x)
    return model

    
def get_squeezenet(optimizer='sgd', lr=0.01, image_size=(224, 224)):
    """
    Method get SqeezeNet v1.1 model, adapt inputs and outputs and return compiled model.
    Reference:
    https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/train_val.prototxt    
    """
    
    inputs = Input((3,)+image_size[::-1])
    base_model = get_base_squeezenet_v1_1(inputs, n_classes=3)
    outputs = base_model.output
    model = Model(input=base_model.input, output=outputs)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.0002, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'precision', 'recall'])
    return model
