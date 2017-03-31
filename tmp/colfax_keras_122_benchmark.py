import os
import sys
import datetime
from time import time
    
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
    

### random data
n_samples = 5000
ndim = 5
inputs = np.random.rand(n_samples, ndim)
targets = (inputs[:, 0] - inputs[:, ndim-1])/(inputs[:, 0] + inputs[:, ndim-1] + 0)

    
### Define a network
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras import backend as K

x_in = Input((ndim,))
x = Dense(100, activation='relu', init='normal')(x_in)
x = Dense(100, activation='relu', init='normal')(x)
x = Dense(50, activation='tanh', init='normal')(x)
x_out = Dense(1)(x)
model = Model(input=x_in, output=x_out)


def K_max_error_percentage(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred)) * 100.0

def max_error_percentage(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred)) * 100.0

model.compile(loss='mae', optimizer='adam', metrics=['mae', K_max_error_percentage])


print("\n {} - Start training".format(datetime.datetime.now()))
start_time = time()
batch_size = 64
history = model.fit(inputs, targets, shuffle=True, batch_size=batch_size, validation_split=0.3, nb_epoch=1000, verbose=2)

elapsed_time = time() - start_time
print("\n Elapsed time: {}".format(elapsed_time))
print("\n {} - Start predictions".format(datetime.datetime.now()))
X_test = np.random.rand(100, ndim)
Y_test = (X_test[:, 0] - X_test[:, ndim-1])/(X_test[:, 0] + X_test[:, ndim-1])
Y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error

args = Y_test.argsort()
y_true = Y_test[args]
y_pred = Y_pred.ravel()[args]
print "Prediction errors: ", mean_absolute_error(Y_test, Y_pred), max_error_percentage(y_true, y_pred)

print("\n {} - End".format(datetime.datetime.now()))
