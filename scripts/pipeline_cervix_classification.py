
# coding: utf-8

# In[1]:

# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[2]:

import os
import sys
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import shutil

# Project
project_common_path = os.path.dirname('.')
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)


# In[3]:

from data_utils import RESOURCES_PATH, GENERATED_DATA, get_annotations
from data_utils import get_id_type_list_from_annotations
from image_utils import get_image_data


# In[4]:

sloth_annotations_filename = os.path.join(RESOURCES_PATH, 'cervix_os.json')
annotations = get_annotations(sloth_annotations_filename)
print("Number of hand-made annotations: ", len(annotations))

trainval_id_type_list = get_id_type_list_from_annotations(annotations, select=['os', 'cervix', 'ok'])
bad_id_type_list = get_id_type_list_from_annotations(annotations, select=['to_remove', ])
print(len(trainval_id_type_list), len(bad_id_type_list))

## Remove green imagery
from data_utils import remove_green_imagery
trainval_id_type_list = remove_green_imagery(trainval_id_type_list)
print(len(trainval_id_type_list))


# In[5]:

import numpy as np
from data_utils import compute_type_distribution
compute_type_distribution(np.array(trainval_id_type_list))


# In[6]:

from xy_providers import DataCache, load_data_cache, save_data_cache
try:
    if cache is None:
        cache_filepath = os.path.join(GENERATED_DATA, 'data_cache.pkl')
        if os.path.exists(cache_filepath):
            print("Load cache from pickle file")
            cache = load_data_cache(cache_filepath)
        else:
            cache = DataCache(0)
except NameError:
    cache_filepath = os.path.join(GENERATED_DATA, 'data_cache.pkl')
    if os.path.exists(cache_filepath):
        print("Load cache from pickle file")
        cache = load_data_cache(cache_filepath)
    else:
        cache = DataCache(0)


# In[7]:

len(cache.cache), len(cache.ids_queue)


# In[8]:

os.environ['KERAS_BACKEND']='tensorflow'

# import tensorflow as tf
# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
# sess = tf.Session(config=config)
# from keras import backend as K
# K.tensorflow_backend.set_session(session=sess)

from custom_mix_cnn_keras_v2 import get_mixed_cnn3


# In[9]:

from keras import backend as K
print(K.backend(), K.image_data_format())


# In[ ]:

from cv_utils import generate_trainval_kfolds
from training_utils import find_best_weights_file2
from data_utils import to_set
from training_utils import classification_train as train, classification_validate as validate
from training_utils import find_best_weights_file2
from training_utils import exp_decay, step_decay


# In[ ]:




# In[ ]:




# In[ ]:




# In[13]:

cnn = get_mixed_cnn3()
cnn.summary()


# In[14]:

from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG

SVG(model_to_dot(cnn, show_shapes=True).create(prog='dot', format='svg'))


# In[ ]:




# In[ ]:




# In[ ]:




# In[33]:

# x = np.arange(nb_epochs)
# plt.plot(exp_decay(x, lr=lr_base, a=a), label='orig')
# plt.plot(exp_decay(x, lr=lr_base, a=a - 0.02), label='mod')
# plt.legend()


# In[14]:

exp_decay(0, lr=0.1,a=0.9, init_epoch=18)


# In[ ]:

# ####################################################
#  Setup NN parameters
# ####################################################

seed = 54321
image_size = (224, 224)

# optimizer = 'nadam_accum'
accum_iters = 16
# nb_epochs = 50
# batch_size = 4
# lr_base = 0.003
# init_epoch = 0
# a = 0.957

optimizer = 'adadelta'
nb_epochs = 50
batch_size = 8
lr_base = 0.1
init_epoch = 18
a = 0.9

load_best_weights = True

# ####################################################
#  Start CV
# ####################################################


n_folds = 4
val_fold_index = 0
val_fold_indices = [2,]

hists = []

for train_id_type_list, val_id_type_list in generate_trainval_kfolds(np.array(trainval_id_type_list), n_folds, seed=seed):
    
    if len(val_fold_indices) > 0:
        if val_fold_index not in val_fold_indices:
            val_fold_index += 1
            continue
        
    save_prefix = 'mixed_cnn3_cervix_class_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
    print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)
    val_fold_index += 1

    print(len(train_id_type_list), len(val_id_type_list))
    assert len(to_set(train_id_type_list) & to_set(val_id_type_list)) == 0, "WTF"
    print(compute_type_distribution(train_id_type_list))
    print(compute_type_distribution(val_id_type_list))

    print("\n {} - Get mixed cnn3 model ...".format(datetime.now()))
    cnn = get_mixed_cnn3(optimizer=optimizer, lr=lr_base, accum_iters=accum_iters)

    if load_best_weights:
        weights_files = glob("weights/%s*.h5" % save_prefix)
        if len(weights_files) > 0:
            best_weights_filename, best_val_loss = find_best_weights_file2(weights_files, field_name='val_loss')
            print("Load best loss weights: ", best_weights_filename, best_val_loss)
            cnn.load_weights(best_weights_filename, by_name=True)

    # lrate_decay_f = lambda epoch: step_decay(epoch, lr=lr_base, base=2.0, period=7)
    lrate_decay_f = lambda epoch: exp_decay(epoch, lr=lr_base, a=a, init_epoch=init_epoch)
    # lrate_decay_f = None
    
    np.random.seed(seed)
    print("\n {} - Start training ...".format(datetime.now()))
    h = train(cnn,
              train_id_type_list,
              val_id_type_list,
              option='cervix', 
              normalize_data=False,
              normalization='',
              nb_epochs=nb_epochs,
              samples_per_epoch=2 * len(train_id_type_list),
              nb_val_samples=len(val_id_type_list),
              lrate_decay_f=lrate_decay_f,
              batch_size=batch_size,
              xy_provider_cache=cache,
              image_size=image_size[::-1],
              seed=seed,              
              save_prefix=save_prefix)    
    if h is None:
        continue
    hists.append(h)
    


# In[ ]:




# In[ ]:




# In[15]:

get_ipython().system(u'ls weights/training*.log')


# In[16]:

import pandas as pd
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')


# In[17]:

df = pd.read_csv('weights/training_mixed_cnn_cervix_class_cvfold=0_opt=adadelta_seed=54321_2017-06-04-16-48.log')
df[['loss', 'val_loss']].plot(ylim=(0.4, 1.0))
df = pd.read_csv('weights/training_mixed_cnn3_cervix_class_cvfold=2_opt=adadelta_seed=54321_2017-06-11-08-51.log')
df[['loss', 'val_loss']].plot(ylim=(0.4, 1.0))


# In[27]:




# In[29]:

df = pd.read_csv('weights/training_mixed_cnn_cervix_class_cvfold=2_opt=adadelta_seed=54321_2017-06-06-08-24.log')
df[['loss', 'val_loss']].plot(ylim=(0.4, 1.0))


# In[30]:

df = pd.read_csv('weights/training_mixed_cnn_cervix_class_cvfold=1_opt=adadelta_seed=54321_2017-06-05-18-44.log')
df[['loss', 'val_loss']].plot(ylim=(0.4, 1.0))


# In[31]:

df = pd.read_csv('weights/training_mixed_cnn_cervix_class_cvfold=0_opt=adadelta_seed=54321_2017-06-04-16-48.log')
df[['loss', 'val_loss']].plot(ylim=(0.4, 1.0))


# In[11]:


import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')



plt.figure(figsize=(12, 4))
ll = len(hists)
for i, h in enumerate(hists):
    plt.subplot(1, ll, i+1)
    plt.plot(h.history['val_loss'], label='val_loss, fold %i' % i)
    plt.plot(h.history['loss'], label='loss, fold %i' % i)   
    plt.legend()


plt.figure(figsize=(12, 4))
ll = len(hists)
for i, h in enumerate(hists):
    plt.subplot(1, ll, i+1)
    plt.plot(h.history['val_categorical_accuracy'], label='val_categorical_accuracy, fold %i' % i)
    plt.plot(h.history['categorical_accuracy'], label='categorical_accuracy, fold %i' % i)   
    plt.legend()
    
# plt.figure(figsize=(12, 4))
# ll = len(hists)
# for i, h in enumerate(hists):
#     plt.subplot(1, ll, i+1)
#     plt.plot(h.history['val_precision'], label='val_precision, fold %i' % i)
#     plt.plot(h.history['precision'], label='precision, fold %i' % i)   
#     plt.legend()
    
# plt.figure(figsize=(12, 4))
# ll = len(hists)
# for i, h in enumerate(hists):
#     plt.subplot(1, ll, i+1)
#     plt.plot(h.history['val_recall'], label='val_recall, fold %i' % i)
#     plt.plot(h.history['recall'], label='recall, fold %i' % i)   
#     plt.legend()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[15]:

from training_utils import find_best_weights_file2
from visu_utils import compute_layer_outputs
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')


# In[16]:

layer_output_f_dict = {}


# In[17]:

layer_names = [
    'block6_sepconv1_act', 
    'block7_sepconv1_act', 
    'block8_sepconv1_act', 
    'block9_sepconv1_act',              
    'block10_sepconv1_act',
    'block11_sepconv1_act',
    'block12_sepconv1_act',
    'block13_sepconv1_act',
    'block14_sepconv2_act',
]


# In[18]:

seed = 54321
optimizer = 'adadelta'
image_size = (224, 224)


# In[19]:

from training_utils import get_train_gen_flow, get_val_gen_flow
from image_utils import scale_percentile

val_fold_index = 0
n_folds = 4
save_prefix = 'mixed_cnn_cervix_class_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)

train_gen, train_flow = get_train_gen_flow(train_id_type_list=trainval_id_type_list,
                                             normalize_data=False,
                                             normalization='',
                                             batch_size=1,
                                             seed=seed,
                                             image_size=image_size,
                                             option='cervix',
                                             save_prefix=save_prefix,
                                             xy_provider_cache=cache,
                                             verbose=1)


# In[15]:

print("\n {} - Get mixed cnn model ...".format(datetime.now()))
cnn = get_mixed_cnn2()

weights_files = glob("weights/%s*.h5" % save_prefix)
if len(weights_files) > 0:
    best_weights_filename, best_val_loss = find_best_weights_file2(weights_files, field_name='val_loss')
    print("Load best loss weights: ", best_weights_filename, best_val_loss)
    cnn.load_weights(best_weights_filename, by_name=True)


# In[16]:

max_counter = 5
for x, y in train_flow:
    layer_outputs = compute_layer_outputs(x, cnn, layer_output_f_dict, layer_names=layer_names)    
    n = 4
    print("Image y = ", y)
    plt.figure(figsize=(12, 4))
    plt.imshow(scale_percentile(x[0, :, :, :]))
    for i, name in enumerate(layer_names):
        img = layer_outputs[name]
        img = np.max(img[0, :, :, :], axis=2)
        if i % n == 0:
            plt.figure(figsize=(12,4))            
        plt.subplot(1, n, i % n + 1)
        plt.imshow(img, interpolation='none')
        plt.title('Layer :' + name)
        
    max_counter -= 1
    if max_counter <= 0:
        break


# In[ ]:




# In[ ]:

# max_counter = 5
# for x, y in train_flow:
#     layer_outputs = compute_layer_outputs(x, cnn, layer_output_f_dict, layer_names=layer_names)    
#     n = 4
#     print("Image y = ", y)
#     plt.figure(figsize=(12, 4))
#     plt.imshow(scale_percentile(x[0, :, :, :]))
#     for i, name in enumerate(layer_names):
#         img = layer_outputs[name]
#         img = np.max(img[0, :, :, :], axis=2)
#         if i % n == 0:
#             plt.figure(figsize=(12,4))            
#         plt.subplot(1, n, i % n + 1)
#         plt.imshow(img, interpolation='none')
#         plt.title('Layer :' + name)
        
#     max_counter -= 1
#     if max_counter <= 0:
#         break


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[11]:

## Cross-validation:
seed = 54321
optimizer = 'adadelta'
image_size = (224, 224)

n_runs = 1
n_folds = 4
seed = 54321
run_counter = 0
cv_mean_losses = 1.1 * np.ones((n_runs, n_folds))
val_fold_indices = [0,]


# In[14]:


while run_counter < n_runs:    
    run_counter += 1
    print("\n\n ---- New run : ", run_counter, "/", n_runs)
    val_fold_index = 0
    for train_id_type_list, val_id_type_list in generate_trainval_kfolds(np.array(trainval_id_type_list), n_folds, seed=seed):

        if len(val_fold_indices) > 0:
            if val_fold_index not in val_fold_indices:
                val_fold_index += 1
                continue
        
        save_prefix = 'mixed_cnn_cervix_class_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
        print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)
        val_fold_index += 1

        print(len(train_id_type_list), len(val_id_type_list))
        assert len(to_set(train_id_type_list) & to_set(val_id_type_list)) == 0, "WTF"
        print(compute_type_distribution(train_id_type_list))
        print(compute_type_distribution(val_id_type_list))

        print("\n {} - Get mixed_cnn model ...".format(datetime.now()))
        cnn = get_mixed_cnn()

        weights_files = glob("weights/%s*.h5" % save_prefix)
        assert len(weights_files) > 0, "Failed to load weights"
        best_weights_filename, best_val_loss = find_best_weights_file2(weights_files, field_name='val_loss')
        print("Load best loss weights: ", best_weights_filename, best_val_loss)
        cnn.load_weights(best_weights_filename)

        loss = validate(cnn, 
                        val_id_type_list, 
                        option='cervix',
                        normalize_data=True,
                        normalization='inception',
                        image_size=image_size[::-1],
                        save_prefix=save_prefix,
                        batch_size=8,
                        seed=seed + run_counter,
                        verbose=1,
                        xy_provider_cache=cache)   
        cv_mean_losses[run_counter-1, val_fold_index-1] = loss
    
print(cv_mean_losses)


# In[45]:

np.mean(cv_mean_losses)


# In[ ]:




# In[12]:

## Predict on test data
from data_utils import test_ids

test_id_type_list = []
for image_id in test_ids:
    test_id_type_list.append((image_id, "Test"))


# In[13]:

from test_utils import classification_predict as predict

predictions = []

n_runs = 1
run_counter = 0
val_fold_indices = [0,]

while run_counter < n_runs:  
    
    run_counter += 1
    print("\n\n ---- New run : ", run_counter, "/", n_runs)
    val_fold_index = 0

    for val_fold_index in range(n_folds):

        if len(val_fold_indices) > 0:
            if val_fold_index not in val_fold_indices:
                val_fold_index += 1
                continue

        save_prefix = 'mixed_cnn_cervix_class_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
        print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)
        val_fold_index += 1

        print("\n {} - Get mixed cnn model ...".format(datetime.now()))
        cnn = get_mixed_cnn()

        weights_files = glob("weights/%s*.h5" % save_prefix)
        assert len(weights_files) > 0, "Failed to load weights"
        best_weights_filename, best_val_loss = find_best_weights_file2(weights_files, field_name='val_loss')
        print("Load best loss weights: ", best_weights_filename, best_val_loss)
        cnn.load_weights(best_weights_filename)

        df = predict(cnn,
                    test_id_type_list, 
                    option='cervix', 
                    normalize_data=True,
                    normalization='inception',
                    image_size=image_size[::-1],
                    save_prefix=save_prefix,
                    batch_size=8,
                    seed=seed,
                    xy_provider_cache=cache)
        predictions.append(df)
    


# In[14]:

import pandas as pd


def compute_mean(predictions):    
    df = predictions[0]
    for p in predictions[1:]:
        df = pd.concat([df, p], axis=0)
    df = df.apply(pd.to_numeric, errors='ignore')        
    gb = df.groupby('image_name')
    df2 = gb.agg(np.mean).reset_index()
    return df2

df = compute_mean(predictions)
df.head()


# In[15]:

from datetime import datetime

info = 'fold=0_mixed_cnn'

now = datetime.now()
sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
sub_file = os.path.join('..', 'results', sub_file)
df.to_csv(sub_file, index=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[100]:

import pandas as pd


def compute_mean(predictions):    
    df = predictions[0]
    for p in predictions[1:]:
        df = pd.concat([df, p], axis=0)
    df = df.apply(pd.to_numeric, errors='ignore')        
    gb = df.groupby('image_name')
    df2 = gb.agg(np.mean).reset_index()
    return df2

df = compute_mean(predictions)
df.head()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



