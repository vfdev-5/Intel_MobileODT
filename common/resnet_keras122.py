from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.backend import set_image_dim_ordering
from keras.optimizers import Adadelta, Nadam, Adam
from keras import __version__

assert __version__ == '1.2.2', "Wrong Keras version : %s" % __version__

set_image_dim_ordering('th')


def get_resnet50(image_size=(224, 224), opt='nadam', lr=0.01):
    """
    Method get ResNet50 model from keras applications, adapt inputs and outputs and return compiled model
    """
    base_model = ResNet50(include_top=False, input_tensor=None, input_shape=(3,) + image_size)
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(input=base_model.input, output=output)

    if opt == 'nadam':
        optimizer = Nadam(lr=lr)
    elif opt == 'adam':
        optimizer = Adam(lr=lr)    
    elif opt == 'adadelta':
        optimizer = Adadelta(lr=lr)
    else:
        raise Exception("Unknown optimizer '%s'" % opt)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', ])
    return model


def get_resnet_original(image_size=(224, 224), opt='nadam', lr=0.01, trained=False, finetunning=True):
    """
    Method get not trained ResNet50 original model from keras applications,
    adapt inputs and outputs and return compiled model
    """
    
    if not trained:
        assert not finetuning, "WTF"

    weights = 'imagenet' if trained else None
    base_model = ResNet50(include_top=False,                          
                          weights=weights, 
                          input_tensor=None, 
                          input_shape=(3,) + image_size)
    
    if finetunning:
        names_to_train=[
            # 'res4e_branch2a', 'res4e_branch2b', 'res4e_branch2c',            
            # 'res4f_branch2a', 'res4f_branch2b', 'res4f_branch2c',
            
            #'res5a_branch2a', 'res5a_branch1', 'res5a_branch2b', 'res5a_branch2c',
            
            #'res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c',
            
            'res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c',            
        ]
        for layer in base_model.layers:
            if layer.name in names_to_train:
                layer.trainable = True
            else:
                layer.trainable = False
    
    x = Flatten()(base_model.output)
    output = Dense(3, activation='softmax', name='output')(x)
    model = Model(input=base_model.input, output=output)
    
    if opt == 'nadam':
        optimizer = Nadam(lr=lr)
    elif opt == 'adam':
        optimizer = Adam(lr=lr)    
    elif opt == 'adadelta':
        optimizer = Adadelta(lr=lr)
    else:
        raise Exception("Unknown optimizer '%s'" % opt)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', ])
    return model
