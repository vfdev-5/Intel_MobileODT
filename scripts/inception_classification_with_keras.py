# coding: utf-8
# Train InceptionV3 model to classify cervix images


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

from inception_keras122 import get_inception
from training_utils import get_trainval_id_type_lists3

print("\n {} - Get train/val lists ...".format(datetime.now()))

train_id_type_list, val_id_type_list = get_trainval_id_type_lists3(val_split=0.15, seed=2017)
print(len(train_id_type_list), len(val_id_type_list))

print("\n {} - Get InceptionV3 model ...".format(datetime.now()))
inception = get_inception(trained=False, finetuning=False)


seed = 2017
np.random.seed(seed)
save_prefix='inception_norm2_cervix_opt=adadelta_seed=%i' % seed

from xy_providers import DataCache
cache = DataCache(0)


if True:
    
    from training_utils import exp_decay
    lr_base = 1.0
    lr_1 = 1.0 * lr_base; a_1 = 0.995
    lrate_decay_f = lambda epoch: exp_decay(epoch, lr=lr_1, a=a_1) 
    
    from training_utils import classification_train as train
      
    nb_epochs = 50
    batch_size = 16
    
    print("\n {} - Start training ...".format(datetime.now()))
    h = train(inception, 
              train_id_type_list, 
              val_id_type_list, 
              option='cervix',
              nb_epochs=nb_epochs,
              samples_per_epoch=2048,
              nb_val_samples=512,
              lrate_decay_f=lrate_decay_f,
              batch_size=batch_size, 
              xy_provider_cache=cache,
              seed=seed,
              save_prefix=save_prefix)

    
#from training_utils import classification_validate as validate
#from test_utils import classification_predict as predict

#batch_size = 4

#print("\n {} - Start validation ...".format(datetime.now()))
#validate(resnet, cervix_val_id_type_list, batch_size=batch_size, xy_provider_cache=cache)

#print("\n {} - Start predictions and write submission ...".format(datetime.now()))
#from test_utils import get_test_id_type_list
#test_id_type_list = get_test_id_type_list()
#cervix_test_id_type_list = [(id_type[0], id_type[1] + '_cervix') for id_type in test_id_type_list]
#predict(resnet, cervix_test_id_type_list, info=save_prefix, batch_size=batch_size)



