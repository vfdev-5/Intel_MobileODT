import numpy as np

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, ZeroPadding2D, Flatten
from keras.layers import Convolution2D, Activation, MaxPooling2D, Lambda
from keras.layers import SeparableConv2D, Add, GlobalAveragePooling2D, AveragePooling2D

from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.optimizers import Adam, Adadelta, SGD

from kfs.optimizers import NadamAccum


vgg_mean_th = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
vgg_mean_tf = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 3))


def vgg_preprocess_th(x):
    x = 255.0 * x - vgg_mean_th
    return x[:, ::-1]  # reverse axis rgb->bgr


def vgg_preprocess_tf(x):
    x = 255.0 * x - vgg_mean_tf
    return x[:, :, :, ::-1]  # reverse axis rgb->bgr


def incv3_preprocess(x):
    x -= 0.5
    x *= 2.0
    return x


def get_conv_dense(optimizer='adam', lr=0.01, n_filters_0=64):
    """
        on 40 epochs
        min train logloss = 1.0581
        min val logloss = 1.0050
    """

    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3, 224, 224)
    else:
        channels_axis = -1
        input_shape = (224, 224, 3)
    n_classes = 3

    inputs = Input(input_shape)
    x = inputs
    x = block_resnet_conv1(x, channels_axis=channels_axis, n_filters=n_filters_0)
    x = Flatten()(x)

    outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "cbnr_pool_1"

    return model


def get_conv_dense2(optimizer='adam', lr=0.01, n_filters_0=64):
    """
        min train logloss =
        min val logloss =
    """

    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3, 224, 224)
    else:
        channels_axis = -1
        input_shape = (224, 224, 3)
    n_classes = 3

    inputs = Input(input_shape)
    x = inputs
    x1 = x
    x1 = block_resnet_conv1(x1, s_id='_1',
                            channels_axis=channels_axis,
                            n_filters=n_filters_0)

    x2 = x
    x2 = MaxPooling2D(pool_size=(5, 5), strides=(4, 4),
                      name='shortcut_pool_1')(x2)

    x = Concatenate(axis=channels_axis)([x1, x2])
    x = Flatten()(x)

    outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "cbnr_pool_1"

    return model


def get_conv_dense3(optimizer='adam', lr=0.01, n_filters_0=64):
    """
        min train logloss = 0.9774
        min val logloss = 0.9745
    """

    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3, 224, 224)
    else:
        channels_axis = -1
        input_shape = (224, 224, 3)
    n_classes = 3

    inputs = Input(input_shape)
    x = inputs

    x1 = x
    x1 = block_resnet_conv1(x1, s_id='_1',
                            channels_axis=channels_axis,
                            n_filters=n_filters_0)
    x2 = x
    x2 = MaxPooling2D(pool_size=(5, 5), strides=(4, 4),
                      name='shortcut_pool_1')(x2)

    x11 = x1
    x11 = block_resnet_conv1(x11, s_id='_2a',
                             channels_axis=channels_axis,
                             n_filters=n_filters_0 * 2)
    x12 = x1
    x12 = MaxPooling2D(pool_size=(5, 5), strides=(4, 4),
                       name='shortcut_pool_2a')(x12)

    x21 = x2
    x21 = block_resnet_conv1(x21, s_id='_2b',
                             channels_axis=channels_axis,
                             n_filters=n_filters_0)
    x22 = x2
    x22 = MaxPooling2D(pool_size=(5, 5), strides=(4, 4),
                       name='shortcut_pool_2b')(x22)

    x = Concatenate(axis=channels_axis)([x11, x12, x21, x22])
    x = Flatten()(x)

    outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "cbnr_pool_1"

    return model


def get_mixed_cnn(optimizer='adam', lr=0.01, accum_iters=8):
    """
        Input
            -> { VGG16(b1,b2,b3); VGG19(b1,b2,b3); ResNet50(c,b2,b3) }
            ->
    """
    if K.image_data_format() == 'channels_first':
        channels_axis = 1
        input_shape = (3, 224, 224)
        vgg_preprocess = vgg_preprocess_th
    else:
        channels_axis = -1
        input_shape = (224, 224, 3)
        vgg_preprocess = vgg_preprocess_tf

    n_classes = 3

    inputs = Input(input_shape)
    x = inputs

    x1_in = x
    x2_in = x
    x3_in = x
    x4_in = x

    # Preprocess for VGG
    # Subtract vgg mean and rgb -> bgr
    x1_in = Lambda(vgg_preprocess,
                   input_shape=input_shape,
                   output_shape=input_shape,
                   name='vgg_preprocess')(x1_in)

    # Preprocess for ResNet
    # Subtract vgg mean and rgb -> bgr
    x2_in = Lambda(vgg_preprocess,
                   input_shape=input_shape,
                   output_shape=input_shape,
                   name='resnet_preprocess')(x2_in)

    # Preprocess for InceptionV3
    x3_in = Lambda(incv3_preprocess,
                   input_shape=input_shape,
                   output_shape=input_shape,
                   name='incv3_preprocess')(x3_in)

    # Preprocess for InceptionV3
    x4_in = Lambda(incv3_preprocess,
                   input_shape=input_shape,
                   output_shape=input_shape,
                   name='xcnn_preprocess')(x4_in)

    vgg19 = VGG19(input_tensor=x1_in, weights='imagenet')
    _rename_model(vgg19, prefix='vgg19_')
    _set_trainable_model(vgg19, value=False)
    l = vgg19.get_layer(name='vgg19_block3_pool')
    l.name = 'vgg19_output'
    x1_out = l.output
    # x1_out.shape = (None, 28, 28, 256)

    resnet = ResNet50(input_tensor=x2_in, weights='imagenet')
    _rename_model(resnet, prefix='resnet_')
    _set_trainable_model(resnet, value=False)
    # resnet_block_3d_act_name = _get_resnet_activation_name(resnet, 'bn3c_branch2c')
    l = resnet.get_layer(index=79)
    l.name = 'resnet_output'
    x2_out = l.output
    # x2_out.shape = (None, 28, 28, 512)

    # Add zero padding to obtain target output layer of size 28 x 28
    x3_in = ZeroPadding2D(padding=(13, 13))(x3_in)
    incv3 = InceptionV3(input_tensor=x3_in, weights='imagenet')
    _rename_model(incv3, prefix='incv3_')
    _set_trainable_model(incv3, value=False)
    l = incv3.get_layer(name='incv3_mixed0')
    l.name = 'incv3_output'
    x3_out = l.output
    # x3_out.shape = (None, 28, 28, 256)

    xcnn = Xception(input_tensor=x4_in, weights='imagenet')
    _rename_model(xcnn, prefix='xcnn_')
    _set_trainable_model(xcnn, value=False)
    l = xcnn.get_layer(index=25)
    l.name = 'xcnn_output'
    x4_out = l.output
    # x4_out.shape = (None, 28, 28, 256)

    x1 = Concatenate(axis=channels_axis)([x1_out, x3_out])
    x2 = Concatenate(axis=channels_axis)([x1_out, x4_out])
    x3 = Concatenate(axis=channels_axis)([x3_out, x4_out])
    x = Add(name='add_mix')([x2_out, x1, x2, x3])

    x = _get_xception_blocks_4_14(x, n_filters=512)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = Dense(n_classes, activation='softmax', name='predictions')(x)

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    elif optimizer == 'nadam_accum':
        opt = NadamAccum(lr=lr, accum_iters=accum_iters)
    else:
        raise Exception("Optimizer '%s' is unknown" % optimizer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['categorical_crossentropy', 'categorical_accuracy'])
    model.name = "mixed_cnn"
    return model


def _set_trainable_model(model, value):
    for l in model.layers:
        l.trainable = value


def _rename_model(model, prefix):
    for l in model.layers:
        l.name = prefix + l.name


def _get_resnet_activation_name(resnet, bn_name_2c):
    t = []
    for l in resnet.layers:
        if l.name == bn_name_2c:
            t.append(l.name)
            continue
        if len(t) > 0:
            t.append(l.name)
        if len(t) == 3:
            break
    return t[-1]


def _get_xception_blocks_4_14(input_layer, n_filters=728):

    x = input_layer

    residual = Convolution2D(n_filters, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(n_filters, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(n_filters, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = Add()([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(n_filters, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(n_filters, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(n_filters, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = Add()([x, residual])
    
    n_filters_1 = int(n_filters * 1.4066)
    n_filters_2 = int(n_filters * 2.11)
    residual = Convolution2D(n_filters_1, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(n_filters, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(n_filters_1, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = Add()([x, residual])

    x = SeparableConv2D(n_filters_2, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(n_filters_1 * 2, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)
    return x


def _get_inception_v3_ending(input_layer, channels_axis):
    x = input_layer
    # mixed 8: 8 x 8 x 1280

    branch3x3 = block_cbnr(x, 192, kernel_size=1, s_id='_m8_1')
    branch3x3 = block_cbnr(branch3x3, 320,
                           kernel_size=3,
                           strides=(2, 2),
                           padding='valid', s_id='_m8_2')

    branch7x7x3 = block_cbnr(x, 192, kernel_size=1, s_id='_m8_3')
    branch7x7x3 = block_cbnr(branch7x7x3, 192, kernel_size=(1, 7), s_id='_m8_4')
    branch7x7x3 = block_cbnr(branch7x7x3, 192, kernel_size=(7, 1), s_id='_m8_5')
    branch7x7x3 = block_cbnr(
        branch7x7x3, 192, kernel_size=3,
        strides=(2, 2), padding='valid', s_id='_m8_6')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Concatenate(axis=channels_axis,
                    name='mixed8')([branch3x3, branch7x7x3, branch_pool])

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = block_cbnr(x, 320, kernel_size=1, s_id='_m%i_1' % (9+i))

        branch3x3 = block_cbnr(x, 384, kernel_size=1, s_id='_m%i_2' % (9+i))
        branch3x3_1 = block_cbnr(branch3x3, 384, kernel_size=(1, 3), s_id='_m%i_3' % (9+i))
        branch3x3_2 = block_cbnr(branch3x3, 384, kernel_size=(3, 1), s_id='_m%i_4' % (9+i))
        branch3x3 = Concatenate(axis=channels_axis,
                                name='mixed9_' + str(i))([branch3x3_1, branch3x3_2])

        branch3x3dbl = block_cbnr(x, 448, kernel_size=1, s_id='_m%i_5' % (9+i))
        branch3x3dbl = block_cbnr(branch3x3dbl, 384, kernel_size=3, s_id='_m%i_6' % (9+i))
        branch3x3dbl_1 = block_cbnr(branch3x3dbl, 384, kernel_size=(1, 3), s_id='_m%i_7' % (9+i))
        branch3x3dbl_2 = block_cbnr(branch3x3dbl, 384, kernel_size=(3, 1), s_id='_m%i_8' % (9+i))
        branch3x3dbl = Concatenate(axis=channels_axis)([branch3x3dbl_1, branch3x3dbl_2])

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = block_cbnr(branch_pool, 192, kernel_size=1, s_id='_m%i_9' % (9+i))
        x = Concatenate(axis=channels_axis,
                        name='mixed' + str(9 + i))([branch1x1, branch3x3, branch3x3dbl, branch_pool])
    return x


def block_resnet_conv1(input_layer, channels_axis, n_filters=64, s_id=''):
    x = ZeroPadding2D((3, 3), name='block' + s_id + '_0pad')(input_layer)
    x = Convolution2D(n_filters,
                      kernel_size=(7, 7), strides=(2, 2),
                      name='block' + s_id + '_conv')(x)
    x = BatchNormalization(axis=channels_axis,
                           name='block' + s_id + '_bn')(x)
    x = Activation('relu', name='block' + s_id + '_act')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2),
                     name='block' + s_id + '_pool')(x)
    return x


def block_cbnr(input_layer,
               n_filters,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               channels_axis=1,
               s_id=''):
    x = Convolution2D(n_filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=False,
                      name='cbnr' + s_id + '_conv')(input_layer)
    x = BatchNormalization(axis=channels_axis,
                           name='cbnr' + s_id + '_bn')(x)
    x = Activation('relu', name='cbnr' + s_id + '_act')(x)
    return x

