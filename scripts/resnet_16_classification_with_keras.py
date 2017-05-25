#
# # coding: utf-8
#
# #
# # To launch on Colfax :
# # qsub -V
# #
#
# # # Trained ResNet-50 classification on os images
# # Test combination:
# #
# # N Samples | ImgAug| Weights | Finetunning | Learning rate | Optimizer | min val loss (< 50 epochs)
# # --- | --- | --- | --- | --- | --- | --- | ---
# # 4200 | 45 deg, 0.05, 0.05 |imagenet | res5b | 1e-5 | Adam | -
#
# import os
# import sys
# from datetime import datetime
# import numpy as np
#
# # Project
# project_common_path = os.path.dirname(__file__)
# project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
# if not project_common_path in sys.path:
#     sys.path.append(project_common_path)
#
# import platform
#
# if 'c001' in platform.node():
#     from colfax_configuration import setup_keras_122
#     setup_keras_122()
#
#
# # In[5]:
#
# from resnet_keras122 import get_resnet_original
#
#
# # In[6]:
#
# from training_utils import get_trainval_id_type_lists3
#
# print("\n {} - Get train/val lists ...".format(datetime.now()))
# train_id_type_list, val_id_type_list = get_trainval_id_type_lists3(n_images_per_class=1400)
# print len(train_id_type_list), len(val_id_type_list)
#
#
# # In[7]:
#
# print("\n {} - Get ResNet-50 model ...".format(datetime.now()))
# resnet = get_resnet_original(opt='adam', lr=1e-5, trained=True, finetunning=True)
#
# # In[8]:
#
# seed = 2017
# np.random.seed(seed)
# save_prefix='resnet50_trained_finetunning_cervix_adam_seed=%i' % seed
#
#
# # In[9]:
#
# from glob import glob
# from training_utils import find_best_weights_file
#
# weights_files = glob("weights/%s*.h5" % save_prefix)
# best_weights_filename, best_val_loss = find_best_weights_file(weights_files)
# print("Best val loss weights: ", best_weights_filename)
#
# if len(best_weights_filename) > 0:
#     # load weights to the model
#     print("Load found weights: ", best_weights_filename)
#     resnet.load_weights(best_weights_filename)
#
#
# from xy_providers import DataCache
# cache = DataCache(0)
#
# if True:
#     from training_utils import exp_decay
#     from training_utils import classification_train as train
#
#     nb_epochs = 50
#     batch_size = 10
#
#     lrate_decay_f = lambda epoch: exp_decay(epoch, lr=lr_1, a=a_1)
#
#     print("\n {} - Start training ...".format(datetime.now()))
#     h = train(resnet,
#               train_id_type_list,
#               val_id_type_list,
#               option='cervix',
#               normalization='resnet',
#               nb_epochs=nb_epochs,
#               samples_per_epoch=1.23 * len(train_id_type_list),
#               nb_val_samples=len(val_id_type_list),
#               lrate_decay_f=None, #lrate_decay_f,
#               batch_size=batch_size,
#               xy_provider_cache=cache,
#               seed=seed,
#               save_prefix=save_prefix)
#
#
# #from training_utils import classification_validate as validate
# #from test_utils import classification_predict as predict
# #batch_size = 4
#
# #print("\n {} - Start validation ...".format(datetime.now()))
# #validate(resnet, val_id_type_list, batch_size=batch_size, xy_provider_cache=cache)
#
#
# # In[44]:
#
# #print("\n {} - Start predictions and write submission ...".format(datetime.now()))
# #from test_utils import get_test_id_type_list
# #test_id_type_list = get_test_id_type_list()
# #os_test_id_type_list = [(id_type[0], id_type[1] + '_os') for id_type in test_id_type_list]
# #predict(resnet, os_test_id_type_list, info=save_prefix, batch_size=batch_size)
#
