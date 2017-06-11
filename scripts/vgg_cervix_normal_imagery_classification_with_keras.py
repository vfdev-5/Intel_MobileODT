
# coding: utf-8

#
# To launch on Colfax : 
# qsub -V resources/vgg_normal_cervix_classification.launch
#

import os
from datetime import datetime
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

# Project
import sys
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)

import platform
if 'c001' in platform.node():
    from colfax_configuration import setup_keras_122
    setup_keras_122()


from vgg16_keras122 import get_vgg16


from data_utils import GENERATED_DATA
print("\n {} - Get train/val lists ...".format(datetime.now()))

trainval_normal_id_type_list = np.load(os.path.join(GENERATED_DATA, 'normal_id_type_list.npz'))['normal_id_type_list']
print len(trainval_normal_id_type_list), trainval_normal_id_type_list[0]

def repeat(id_type_list, output_size):
    n = int(np.ceil(output_size * 1.0 / len(id_type_list)))    
    out = np.tile(id_type_list, [n, 1])
    return out[:output_size]


def generate_trainval_kfolds(id_type_list, n_folds, seed):
    
    types = (('Type_1', 'AType_1'), ('Type_2', 'AType_2'), ('Type_3', 'AType_3'))
    out = [None, None, None]
    for i, ts in enumerate(types):
        o = id_type_list[(id_type_list[:, 1] == ts[0]) | (id_type_list[:, 1] == ts[1])]
        out[i] = o

    ll = max([len(o) for o in out])
    out = np.array([repeat(o, ll) for o in out])
    out = out.reshape((3 * ll, 2))  
    np.random.seed(seed)
    np.random.shuffle(out)

    for val_fold_index in range(n_folds):
        ll = len(out)
        size = int(ll * 1.0 / n_folds + 1.0)
        overlap = (size * n_folds - ll) * 1.0 / (n_folds - 1.0)
        val_start = int(round(val_fold_index * (size - overlap)))
        val_end = val_start + size

        val_id_type_list = out[val_start:val_end]    
        train_id_type_list = np.array([
                [i[0], i[1]] for i in out if np.sum(np.prod(i == val_id_type_list, axis=1)) == 0
        ])
        yield train_id_type_list, val_id_type_list
        
        
def compute_type_distribution(id_type_list):
    types = (('Type_1', 'AType_1'), ('Type_2', 'AType_2'), ('Type_3', 'AType_3'))
    ll = len(id_type_list)
    out = [0.0, 0.0, 0.0]
    for i, ts in enumerate(types):
        for t in ts:
            out[i] += (id_type_list[:, 1] == t).sum()        
        out[i] *= 1.0 / ll
    return out

def to_set(id_type_array):
    return set([(i[0], i[1]) for i in id_type_array.tolist()])



from xy_providers import DataCache
try:
    if cache is None:
        cache = DataCache(0)
except NameError:
    cache = DataCache(0)
    
    
from training_utils import exp_decay, step_decay
from training_utils import find_best_weights_file
from imagery_classification.normal_types_classification import classification_train as train

seed = 2017

optimizer = 'adam'
lr_base = 0.00001
names_to_train=[            
    #'block3_conv1', 'block3_conv2', 'block3_conv3',
    #'block4_conv1', 'block4_conv2', 'block4_conv3',
    #'block5_conv1', 'block5_conv2',    
    'block5_conv3',
]
nb_epochs = 50
batch_size = 8

load_best_weights = False

# Iterate over folds
n_folds = 7
val_fold_index = 0

for  train_id_type_list, val_id_type_list in generate_trainval_kfolds(trainval_normal_id_type_list, n_folds, seed=seed):

    save_prefix='vgg16_finetunning_b5c3_cervix_normal_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
    print "\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds
    val_fold_index += 1

    print(len(train_id_type_list), len(val_id_type_list))
    assert len(to_set(train_id_type_list) & to_set(val_id_type_list)) == 0, "WTF"
    print compute_type_distribution(train_id_type_list) 
    print compute_type_distribution(val_id_type_list)

    print("\n {} - Get VGG16 model ...".format(datetime.now()))
    vgg = get_vgg16(trained=True, finetuning=True, optimizer=optimizer, names_to_train=names_to_train, lr=lr_base, image_size=(299, 299))
    vgg.trainable_weights
    
    if load_best_weights:
        weights_files = glob("weights/%s*.h5" % save_prefix)
        if len(weights_files) > 0:
            best_weights_filename, best_val_loss = find_best_weights_file2(weights_files, field_name='val_loss')
            print "Load best loss weights: ", best_weights_filename, best_val_loss    
            vgg.load_weights(best_weights_filename)
        
#     lrate_decay_f = lambda epoch: step_decay(epoch, lr=lr_base, base=2.0, period=7) 
    lrate_decay_f = lambda epoch: exp_decay(epoch, lr=lr_base, a=0.935) 
    np.random.seed(seed)
    print("\n {} - Start training ...".format(datetime.now()))
    h = train(vgg, 
              train_id_type_list, 
              val_id_type_list, 
              option='cervix',
              normalization='vgg',
              nb_epochs=nb_epochs,
              samples_per_epoch=1.0 * len(train_id_type_list),
              nb_val_samples=len(val_id_type_list),
              lrate_decay_f=lrate_decay_f,
              batch_size=batch_size,
              xy_provider_cache=cache,
              image_size=(299, 299),
              seed=seed,
              save_prefix=save_prefix)
    if h is None:
        break

print "\n\n------- END of CV ------\n\n" 