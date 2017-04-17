
# coding: utf-8

# # Trained ResNet-50 classification on cervix images


import os
import sys
from datetime import datetime
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

from resnet_keras122 import get_resnet_original
from training_utils import get_trainval_id_type_lists

print("\n {} - Get train/val lists ...".format(datetime.now()))
train_id_type_list, val_id_type_list = get_trainval_id_type_lists()

cervix_train_id_type_list = [(id_type[0], id_type[1] + '_cervix') for id_type in train_id_type_list if id_type[0] != '1339']
cervix_val_id_type_list = [(id_type[0], id_type[1] + '_cervix') for id_type in val_id_type_list if id_type[0] != '1339']


# In[9]:

print("\n {} - Get ResNet-50 model ...".format(datetime.now()))
resnet = get_resnet_original(opt='adadelta')


# In[11]:

seed = 2017
np.random.seed(seed)
save_prefix='resnet_not_trained_original_cervix_adadelta_seed=%i' % seed


# In[12]:

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


# In[14]:

from xy_providers import DataCache
cache = DataCache(2000)


# In[ ]:

if True:
    from training_utils import classification_train as train
      
    nb_epochs = 50
    batch_size = 4
    
    print("\n {} - Start training ...".format(datetime.now()))
    h = train(resnet, 
              cervix_train_id_type_list, 
              cervix_val_id_type_list, 
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
validate(resnet, cervix_val_id_type_list, batch_size=batch_size, xy_provider_cache=cache)

print("\n {} - Start predictions and write submission ...".format(datetime.now()))
from test_utils import get_test_id_type_list
test_id_type_list = get_test_id_type_list()
cervix_test_id_type_list = [(id_type[0], id_type[1] + '_cervix') for id_type in test_id_type_list]
predict(resnet, cervix_test_id_type_list, info=save_prefix, batch_size=batch_size)



