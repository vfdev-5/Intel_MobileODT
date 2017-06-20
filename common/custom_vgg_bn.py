
import numpy as np

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Concatenate
from keras.layers import Convolution2D, Activation, MaxPooling2D
from keras.optimizers import Adam, Adadelta, SGD
from keras.applications.vgg16 import VGG16

from keras_metrics import precision, recall


def get_custom_vgg_bn(optimizer='adam', lr=0.01):
    """
    Custom vgg network with two inputs (cervix/os)

    No need to normalize input data. It is done in cnn operations

    input1 -> vgg_convs (blocks 1-3 trained on imagenet)->\__concat_block4,5__FC blocks
    input2 -> vgg_convs (blocks 1-2 trained on imagenet)->/

    """

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        input_shape1 = (3, 224, 224)
        input_shape2 = (3, 112, 112)
    else:
        channel_axis = -1
        input_shape1 = (224, 224, 3)
        input_shape2 = (112, 112, 3)
    n_classes = 3

    vgg_convs_model1 = _get_vgg_conv_model(input_shape1, 'imagenet', output_layer_name='block3_pool', name_suffix='_1')
    vgg_convs_model2 = _get_vgg_conv_model(input_shape2, 'imagenet', output_layer_name='block2_pool', name_suffix='_2')
    
    #for layer in vgg_convs_model1.layers:
    #    if layer.name in ['block5_conv3_1', ]:
    #        layer.trainable = True

    #for layer in vgg_convs_model2.layers:
    #    if layer.name in ['block4_conv3_1', ]:
    #        layer.trainable = True
    
    x1 = vgg_convs_model1.outputs[0]
    x2 = vgg_convs_model2.outputs[0]
        
    x = Concatenate(axis=channel_axis)([x1, x2])
    # Shape : 256+128 x 28 x 28

    # Block 4: 
    x = block_conv(x, 384, s_id=4, channel_axis=channel_axis)
    # Block 5: 
    x = block_conv(x, 512, s_id=5, channel_axis=channel_axis)
    # Block 6: 
    x = block_conv(x, 512, s_id=6, channel_axis=channel_axis)

    
    x = Flatten()(x)
    x = Dense(1024, name='fc1')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('elu')(x)    
    x = Dense(1024, name='fc2')(x)
    x = Activation('elu')(x)        
    x = Dense(n_classes, activation='softmax', name='predictions')(x)

    in1 = vgg_convs_model1.inputs[0]
    in2 = vgg_convs_model2.inputs[0]
    model = Model(inputs=[in1, in2], outputs=x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "Custom_double_VGG16"

    return model


vgg_mean_th = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
vgg_mean_tf = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 3))


def vgg_preprocess_th(x):
    x = 255.0 * x - vgg_mean_th
    return x[:, ::-1]  # reverse axis rgb->bgr


def vgg_preprocess_tf(x):
    x = 255.0 * x - vgg_mean_tf
    return x[:, :, :, ::-1]  # reverse axis rgb->bgr


def block_conv(input_layer, n_filters, s_id, channel_axis):
   
    x = Convolution2D(filters=n_filters,
                      kernel_size=(1, 3),
                      padding='same',
                      name='block%i_conv1' % s_id)(input_layer)
    x = BatchNormalization(axis=channel_axis, name='block%i_bn1' % s_id)(x)
    x = Activation('elu', name='block%i_elu1' % s_id)(x)    
        
    x = Convolution2D(filters=n_filters,
                      kernel_size=(3, 1),
                      padding='same',
                      name='block%i_conv2' % s_id)(x)
    x = BatchNormalization(axis=channel_axis, name='block%i_bn2' % s_id)(x)
    x = Activation('elu', name='block%i_elu2' % s_id)(x)     

    x = Convolution2D(filters=n_filters,
                      kernel_size=(3, 3),
                      padding='same',
                      name='block%i_conv3' % s_id)(x)
    x = BatchNormalization(axis=channel_axis, name='block%i_bn3' % s_id)(x)
    x = Activation('elu', name='block%i_elu3' % s_id)(x)    
    
    x = MaxPooling2D(pool_size=(2, 2),
                      name='block%i_pool' % s_id)(x)
    return x


def _get_vgg_conv_model(input_shape, weights='imagenet', output_layer_name='block5_pool', name_suffix=''):

    inputs = Input(shape=input_shape, name='input')
    x = inputs

    if K.image_data_format() == 'channels_first':
        vgg_preprocess = vgg_preprocess_th
    else:
        vgg_preprocess = vgg_preprocess_tf

    # Subtract vgg mean and rgb -> bgr
    x = Lambda(vgg_preprocess,
               input_shape=input_shape, output_shape=input_shape,
               name='vgg_preprocess')(x)

    vgg = VGG16(include_top=False, weights=weights, input_tensor=x)
    
    # Rename layers:
    for layer in vgg.layers:
        layer.name += name_suffix
        layer.trainable = False

    model = Model(inputs=vgg.inputs, outputs=vgg.get_layer(name=output_layer_name + name_suffix).output)        
    return model
