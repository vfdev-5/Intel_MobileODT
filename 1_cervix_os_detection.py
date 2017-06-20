import os
import sys
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np

# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)

from data_utils import GENERATED_DATA
from data_utils import test_ids
from xy_providers import DataCache

from unet_keras_v1 import get_unet
from training_utils import find_best_weights_file2
from test_utils import segmentation_predict as predict
import cv2
from data_utils import get_filename
from image_utils import imwrite, get_image_data
from postprocessing_utils import os_cervix_postproc, os_cervix_merge_masks, get_bbox


cache = DataCache(0)
test_id_type_list = []
for image_id in test_ids:
    test_id_type_list.append((image_id, "Test"))

# ####################################################
#  Crop to cervix/os
# ####################################################

seed = 54321
optimizer = 'adadelta'
input_shape = (3, 224, 224)

n_folds = 6
batch_size = 8

for val_fold_index in range(n_folds):

    save_prefix = 'unet_os_cervix_detector_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
    print("\n {} - Get UNET model ...".format(datetime.now()))
    unet = get_unet(input_shape=input_shape, n_classes=2, optimizer=None)

    weights_files = glob(os.path.join(GENERATED_DATA, "%s*.h5" % save_prefix))
    if len(weights_files) > 0:
        best_weights_filename, best_val_loss = find_best_weights_file2(weights_files, field_name='val_loss')
        print("Load best loss weights: ", best_weights_filename, best_val_loss)
        unet.load_weights(best_weights_filename)
    else:
        continue

    print("\n {} - Start predictions ...".format(datetime.now()))

    predict(unet,
            test_id_type_list,
            save_prefix=save_prefix,
            batch_size=batch_size,
            xy_provider_cache=cache,
            image_size=input_shape[1:][::-1])


no_os_detected_id_type_list = []
no_cervix_detected_id_type_list = []
image_size = (299, 299)

id_type_list = list(test_id_type_list)

l = len(id_type_list)
for i, (image_id, image_type) in enumerate(id_type_list):

    print("-- %i / %i | %s, %s " % (i, l, image_id, image_type))

    filename = get_filename(image_id + '_' + image_type, 'os_cervix_bbox')
    if os.path.exists(filename):
        continue

    os_cervix_masks = []
    for val_fold_index in range(n_folds):
        save_prefix = 'unet_os_cervix_detector_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
        p = os.path.join(GENERATED_DATA, 'os_cervix_label__%s' % save_prefix)
        if os.path.exists(p):
            os_cervix_masks.append(get_image_data(image_id + "_" + image_type, 'os_cervix_label__%s' % save_prefix))

    final_os_cervix_mask = os_cervix_masks[0]
    for mask in os_cervix_masks[1:]:
        final_os_cervix_mask = os_cervix_merge_masks(final_os_cervix_mask, mask)
    final_os_cervix_mask = os_cervix_postproc(final_os_cervix_mask)

    os_cervix = cv2.resize(final_os_cervix_mask, dsize=image_size, interpolation=cv2.INTER_NEAREST)
    cervix_mask = os_cervix[:, :, 1]
    if np.sum(cervix_mask) > 0:
        cervix_bbox = get_bbox(cervix_mask)

        os_mask = os_cervix[:, :, 0]
        if np.sum(os_mask) > 0:
            os_bbox = get_bbox(os_mask)
        else:
            # set cervix instead of os
            print("No os detected on the image: %s, %s" % (image_id, image_type))
            no_os_detected_id_type_list.append((image_id, image_type))
            os_bbox = cervix_bbox

        _image_id = image_id + '_' + image_type
        imwrite(final_os_cervix_mask, _image_id, 'os_cervix_label_final')
        np.savez(filename, os_bbox=os_bbox, cervix_bbox=cervix_bbox, image_size=image_size)
    else:
        print("No cervix detected on the image: %s, %s" % (image_id, image_type))
        no_cervix_detected_id_type_list.append((image_id, image_type))

print("List of images, where no cervix was detected: ", no_cervix_detected_id_type_list)
print("List of images, where no os was detected: ", no_os_detected_id_type_list)



