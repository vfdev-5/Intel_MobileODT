import os
import sys
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import shutil

# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)

from data_utils import RESOURCES_PATH, GENERATED_DATA, get_annotations
from data_utils import get_id_type_list_from_annotations
from image_utils import generate_label_images
from cv_utils import generate_trainval_kfolds
from xy_providers import DataCache


# ####################################################
#  Transform `sloth` annotations to label images
# ####################################################

sloth_annotations_filename = os.path.join(RESOURCES_PATH, 'cervix_os.json')
annotations = get_annotations(sloth_annotations_filename)
print("Number of hand-made annotations: ", len(annotations))


target_files_path = os.path.join(GENERATED_DATA, "trainval_labels_0")
n_target_files = 0
if os.path.exists(target_files_path):
    n_target_files = len(glob(os.path.join(target_files_path, "*.npz")))

if n_target_files != len(annotations):
    print("\n {} - Generated target files".format(datetime.now()))
    shutil.rmtree(target_files_path)
    generate_label_images(annotations)
else:
    print("Found %i generated target files" % n_target_files)

trainval_id_type_list = get_id_type_list_from_annotations(annotations, select=['os', 'cervix'])

print("\n {} - Cross-validation training U-Net model ...".format(datetime.now()))
cache = DataCache(0)

# ####################################################
#  Load resized images to cache and remove green imagery
# ####################################################
import numpy as np
from xy_providers import cached_image_provider as xy_provider
from imagery_classification.green import is_green_type

channels_first = True
gen = xy_provider(trainval_id_type_list, image_size=(299, 299), channels_first=channels_first, cache=cache)

new_trainval_id_type_list = []
for x, _, info in gen:
    img = (255.0 * x).astype(np.uint8)
    if channels_first:
        img = img.transpose([1, 2, 0])

    if not is_green_type(img):
        new_trainval_id_type_list.append(info)


# ####################################################
#  Setup NN parameters
# ####################################################
from unet_keras_v1 import get_unet
from training_utils import segmentation_train as train, segmentation_validate as validate
from data_utils import compute_type_distribution


def to_set(id_type_array):
    return set([(i[0], i[1]) for i in id_type_array.tolist()])

seed = 54321
optimizer = 'adadelta'
input_shape = (3, 224, 224)

nb_epochs = 100
batch_size = 8
lr_base = 1.0

load_best_weights = False


# ####################################################
#  Start CV
# ####################################################
from training_utils import find_best_weights_file2
from training_utils import exp_decay, step_decay

n_folds = 6
val_fold_index = 0

for train_id_type_list, val_id_type_list in generate_trainval_kfolds(trainval_id_type_list, n_folds, seed=seed):

    save_prefix = 'unet_os_cervix_detector_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
    print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)
    val_fold_index += 1

    print(len(train_id_type_list), len(val_id_type_list))
    assert len(to_set(train_id_type_list) & to_set(val_id_type_list)) == 0, "WTF"
    print(compute_type_distribution(train_id_type_list))
    print(compute_type_distribution(val_id_type_list))

    print("\n {} - Get UNET model ...".format(datetime.now()))
    unet = get_unet(input_shape=input_shape, n_classes=2, optimizer=optimizer)

    if load_best_weights:
        weights_files = glob("weights/%s*.h5" % save_prefix)
        if len(weights_files) > 0:
            best_weights_filename, best_val_loss = find_best_weights_file2(weights_files, field_name='val_loss')
            print("Load best loss weights: ", best_weights_filename, best_val_loss)
            unet.load_weights(best_weights_filename)

    # lrate_decay_f = lambda epoch: step_decay(epoch, lr=lr_base, base=2.0, period=7)
    lrate_decay_f = lambda epoch: exp_decay(epoch, lr=lr_base, a=0.967)
    # lrate_decay_f = None

    np.random.seed(seed)
    print("\n {} - Start training ...".format(datetime.now()))
    h = train(unet,
              train_id_type_list,
              val_id_type_list,
              nb_epochs=nb_epochs,
              samples_per_epoch=1.0 * len(train_id_type_list),
              nb_val_samples=len(val_id_type_list),
              lrate_decay_f=lrate_decay_f,
              batch_size=batch_size,
              xy_provider_cache=cache,
              image_size=input_shape[1:][::-1],
              seed=seed,
              save_prefix=save_prefix)
    if h is None:
        break




