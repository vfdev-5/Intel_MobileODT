import numpy as np

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Convolution2D
from keras.layers import Input, Lambda
from keras.optimizers import Adadelta, Adam
from keras import backend as K


def get_vgg16(trained=True, finetuning=True, optimizer='adadelta', lr=0.01, image_size=(224, 224), names_to_train=None):
    """
    Method get VGG16 model from keras applications, adapt inputs and outputs and return compiled model
    """
    if not trained:
        assert not finetuning, "WTF"

    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3,) + image_size[::-1]
        vgg_preprocess = vgg_preprocess_th
    else:
        channels_axis = -1
        input_shape = image_size[::-1] + (3,)
        vgg_preprocess = vgg_preprocess_tf        
        
    inputs = Input(shape=input_shape, name='input')
    x = inputs

    # Subtract vgg mean and rgb -> bgr
    x = Lambda(vgg_preprocess,
               input_shape=input_shape, output_shape=input_shape,
               name='vgg_preprocess')(x)
    
        
    n_classes = 3
    weights = 'imagenet' if trained else None
    base_model = VGG16(include_top=False, input_tensor=x, weights=weights)

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

    x = base_model.outputs[0]
    x = block_conv(x, 512, 6, channels_axis)
                
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    #x = BatchNormalization(axis=channels_axis)(x)
    #x = Activation('elu')(x)    
    x = Dense(1024, activation='relu', name='fc2')(x)
    #x = BatchNormalization(axis=channels_axis)(x)    
    #x = Activation('elu')(x)        
    x = Dense(n_classes, activation='softmax', name='predictions')(x)
    
    outputs = Dense(3, activation='softmax', name='output')(x)
    model = Model(inputs=base_model.inputs, outputs=outputs)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_crossentropy', 'categorical_accuracy'])
    return model


vgg_mean_th = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
vgg_mean_tf = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 3))


def vgg_preprocess_th(x):
    x = 255.0 * x - vgg_mean_th
    return x[:, ::-1]  # reverse axis rgb->bgr


def vgg_preprocess_tf(x):
    x = 255.0 * x - vgg_mean_tf
    return x[:, :, :, ::-1]  # reverse axis rgb->bgr


def block_conv(input_layer, n_filters, s_id, channels_axis):
   
    x = Convolution2D(filters=n_filters,
                      kernel_size=(1, 3),
                      padding='same',
                      activation='relu',
                      name='block%i_conv1' % s_id)(input_layer)
    #x = BatchNormalization(axis=channels_axis, name='block%i_bn1' % s_id)(x)
    #x = Activation('elu', name='block%i_elu1' % s_id)(x)    
        
    x = Convolution2D(filters=n_filters,
                      kernel_size=(3, 1),
                      padding='same',
                      activation='relu',                      
                      name='block%i_conv2' % s_id)(x)
    #x = BatchNormalization(axis=channels_axis, name='block%i_bn2' % s_id)(x)
    #x = Activation('elu', name='block%i_elu2' % s_id)(x)     

    x = Convolution2D(filters=n_filters,
                      kernel_size=(3, 3),
                      padding='same',
                      activation='relu',                      
                      name='block%i_conv3' % s_id)(x)
    #x = BatchNormalization(axis=channels_axis, name='block%i_bn3' % s_id)(x)
    #x = Activation('elu', name='block%i_elu3' % s_id)(x)    
    
    x = MaxPooling2D(pool_size=(2, 2),
                      name='block%i_pool' % s_id)(x)
    return x
