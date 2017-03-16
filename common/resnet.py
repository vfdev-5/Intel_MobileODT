from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model

from keras.backend import set_image_dim_ordering, image_dim_ordering
set_image_dim_ordering('th')


def get_resnet50(image_size=(224, 224)):
    """
    Method get ResNet50 model from keras applications, adapt inputs and outputs and return compiled model
    """
    base_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(3,) + image_size)
    x = Flatten()(base_model.output)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy',])
    return model
