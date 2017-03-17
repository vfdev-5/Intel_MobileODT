import os
import datetime

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.callbacks import ModelCheckpoint

# Project
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common'))
from data_utils import type_1_ids, type_2_ids, type_3_ids, test_ids
from data_utils import get_filename
from training_utils import get_trainval_id_type_lists, get_test_id_type_list, data_iterator
from metrics import logloss_mc
from unet_keras122 import get_unet


print("\n=========================")
print("Training dataset: ")
print("- type 1: ", len(type_1_ids))
print("- type 2: ", len(type_2_ids))
print("- type 3: ", len(type_3_ids))

print("Test dataset: ")
print("- ", len(test_ids))
print("=========================\n")


def train(model, train_id_type_list, val_id_type_list, batch_size=16, nb_epochs=10, image_size=(224, 224)):
    samples_per_epoch = 2048
    nb_val_samples = 512

    if not os.path.exists('weights'):
        os.mkdir('weights')

    weights_filename = os.path.join("weights", "unet_simple.h5")
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True)

    print("Training parameters: ", batch_size, nb_epochs, samples_per_epoch, nb_val_samples)
    
    train_iter = data_iterator(train_id_type_list, batch_size=batch_size, image_size=image_size, verbose=1)
    val_iter = data_iterator(val_id_type_list, batch_size=batch_size, image_size=image_size, verbose=1)
    
    history = model.fit_generator(
        train_iter,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epochs,
        validation_data=val_iter,
        nb_val_samples=nb_val_samples,
        callbacks=[model_checkpoint],
        verbose=1,
    )

    return history


def validate(model, val_id_type_list, batch_size=16, image_size=(224, 224)):
    val_iter = data_iterator(val_id_type_list, batch_size=batch_size, image_size=image_size, test_mode=True)

    total_loss = 0.0
    total_counter = 0 
    for X, Y_true, _ in val_iter:           
        s = Y_true.shape[0]
        total_counter += s
        Y_pred = model.predict(X)
        loss = logloss_mc(Y_true, Y_pred)
        total_loss += s * loss
        print("--", total_counter, "batch loss : ", loss)

    if total_counter == 0:
        total_counter += 1

    total_loss *= 1.0 / total_counter   
    print("Total loss : ", total_loss)
    
    
def predict(model, batch_size=16, image_size=(224, 224), info=''):

    test_id_type_list = get_test_id_type_list()
    test_iter = data_iterator(test_id_type_list, batch_size=batch_size, image_size=image_size, test_mode=True)
    
    df = pd.DataFrame(columns=['image_name','Type_1','Type_2','Type_3'])
    total_counter = 0
    for X, _, image_ids in test_iter:            
        Y_pred = model.predict(X)    
        s = X.shape[0]
        total_counter += s
        print("--", total_counter)
        for i in range(s):
            df.loc[total_counter + i, :] = (image_ids[i] + '.jpg', ) + tuple(Y_pred[i, :])

    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    df.to_csv(sub_file, index=False)


if __name__ == "__main__":

    import platform

    batch_size = 16
    if 'c001' in platform.node():
        batch_size = 512
        print("-- On the cluster --")

    print("\n {} - Get train/val lists ...".format(datetime.datetime.now()))
    train_id_type_list, val_id_type_list = get_trainval_id_type_lists()
    print("\n {} - Get U-Net model ...".format(datetime.datetime.now()))
    unet = get_unet()
    print("\n {} - Start training ...".format(datetime.datetime.now()))
    train(unet, train_id_type_list, val_id_type_list, nb_epochs=1, batch_size=batch_size)
    print("\n {} - Start validation ...".format(datetime.datetime.now()))
    # validate(unet, val_id_type_list, batch_size=batch_size)
    print("\n {} - Start predictions and write submission ...".format(datetime.datetime.now()))
    # predict(unet, info='unet_no_additional', batch_size=batch_size)
    print("\n {} - Scripted finished".format(datetime.datetime.now()))





