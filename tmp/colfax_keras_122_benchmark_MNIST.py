import os
import sys
import datetime
    
import numpy as np

# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)


import platform

if 'c001' in platform.node():
    from colfax_configuration import setup_keras_122
    setup_keras_122()
    

### MNIST data
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()


y_train = to_categorical(y_train, num_classes=None)
y_test = to_categorical(y_test, num_classes=None)


### Define a network
from keras.models import Model
from keras.layers import Dense, Dropout, Input

x_in = Input((ndim,))
x = Dense(100, activation='relu', init='normal')(x_in)
x = Dropout(0.5)(x)
x = Dense(100, activation='relu', init='normal')(x)
x = Dropout(0.5)(x)
x_out = Dense(10, activation='softmax')(x)
model = Model(input=x_in, output=x_out)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print("\n {} - Start training".format(datetime.datetime.now()))
batch_size = 32
history = model.fit(x=x_train, y=y_train, 
                    shuffle=True, batch_size=batch_size, 
                    validation_split=0.3, 
                    nb_epoch=100, verbose=2)

print("\n {} - Start predictions".format(datetime.datetime.now()))
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score

args = Y_test.argsort()
y_true = Y_test[args]
y_pred = Y_pred.ravel()[args]
print "Prediction errors: ", accuracy_score(Y_test, Y_pred)

print("\n {} - End".format(datetime.datetime.now()))
