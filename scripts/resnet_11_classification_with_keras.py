
# coding: utf-8

# # Trained ResNet-50 classification 
# 
# - new data generators

# In[2]:

import os
from datetime import datetime
import numpy as np


# In[3]:

# Project
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname('.')), '..', 'common'))

from resnet_keras122 import get_resnet50
from training_utils import get_trainval_id_type_lists

print("\n {} - Get train/val lists ...".format(datetime.now()))
train_id_type_list, val_id_type_list = get_trainval_id_type_lists()

print("\n {} - Get ResNet-50 model ...".format(datetime.now()))
resnet = get_resnet50(opt='adadelta')
resnet.summary()

seed = 2017
np.random.seed(seed)
save_prefix='resnet_4_dense_adadelta_seed=%i' % seed

from glob import glob

weights_files = glob("weights/%s*.h5" % save_prefix)
best_val_loss = 1e5
best_weights_filename = ""
for f in weights_files:
    index = os.path.basename(f).index('-')
    loss = float(os.path.basename(f)[index+1:-4])
    if best_val_loss > loss:
        best_val_loss = loss
        best_weights_filename = f
print("Best val loss weights: ", best_weights_filename)


if len(best_weights_filename) > 0:
    # load weights to the model
    print("Load found weights: ", best_weights_filename)
    resnet.load_weights(best_weights_filename)

from xy_providers import DataCache
cache = DataCache(2000)

if True:
    from training_utils import classification_train as train
      
    nb_epochs = 50
    batch_size = 4
    
    print("\n {} - Start training ...".format(datetime.now()))
    h = train(resnet, 
              train_id_type_list, 
              val_id_type_list, 
              nb_epochs=nb_epochs,
              lrate_decay_f=None,
              batch_size=batch_size, 
              xy_provider_cache=cache,
              seed=seed,
              save_prefix=save_prefix)


from training_utils import classification_validate as validate
from test_utils import classification_predict as predict

batch_size = 4


print("\n {} - Start validation ...".format(datetime.now()))
validate(resnet, val_id_type_list, batch_size=batch_size, xy_provider_cache=cache)


print("\n {} - Start predictions and write submission ...".format(datetime.now()))
from test_utils import get_test_id_type_list
test_id_type_list = get_test_id_type_list()
predict(resnet, test_id_type_list, info=save_prefix, batch_size=batch_size)


