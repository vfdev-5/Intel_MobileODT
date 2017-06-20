import os
import sys
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import pandas as pd


# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)

from xy_providers import DataCache
from data_utils import test_ids, GENERATED_DATA
from test_utils import classification_predict as predict
from custom_squeezenet import get_cnn
from training_utils import find_best_weights_file2


cache = DataCache(0)


# ####################################################
#  Start predictions
# ####################################################


test_id_type_list = []
for image_id in test_ids:
    test_id_type_list.append((image_id, "Test"))

seed = 54321
optimizer = 'adadelta'
image_size = (299, 299)
batch_size = 8

n_folds = 5
predictions = []
run_counter = 0
n_runs = 2

while run_counter < n_runs:
    run_counter += 1
    print("\n\n ---- New run : ", run_counter, "/", n_runs)

    # SqueezeNet on 5 folds
    for val_fold_index in range(n_folds):
        save_prefix = 'cnn_os_cervix_class_cvfold=%i_opt=%s_seed=%i' % (val_fold_index, optimizer, seed)
        print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)
        val_fold_index += 1

        print("\n {} - Get custom squeezenet model ...".format(datetime.now()))
        cnn = get_cnn()

        weights_files = glob(os.path.join(GENERATED_DATA, "%s*.h5" % save_prefix))
        assert len(weights_files) > 0, "Failed to load weights"
        best_weights_filename, best_val_loss = find_best_weights_file2(weights_files, field_name='val_loss')
        print("Load best loss weights: ", best_weights_filename, best_val_loss)
        cnn.load_weights(best_weights_filename)

        df = predict(cnn,
                     test_id_type_list,
                     option='cervix/os',
                     normalize_data=True,
                     normalization='vgg',
                     image_size=image_size[::-1],
                     save_prefix=save_prefix,
                     batch_size=batch_size,
                     seed=seed + run_counter,
                     xy_provider_cache=cache)
        predictions.append(df)


def compute_mean(predictions):
    df = predictions[0]
    for p in predictions[1:]:
        df = pd.concat([df, p], axis=0)
    df = df.apply(pd.to_numeric, errors='ignore')
    gb = df.groupby('image_name')
    df2 = gb.agg(np.mean).reset_index()
    return df2


df = compute_mean(predictions)

info = 'mean_cv=6_squeezenet'
now = datetime.now()
sub_file = 'submission_' + info + '.csv'
sub_file = os.path.join('results', sub_file)
df.to_csv(sub_file, index=False)

