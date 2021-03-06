import platform

if 'c001' in platform.node():
    import sys
    # Remove Keras 1.1
    sys.path = sys.path[2:]

import os
import datetime

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.callbacks import ModelCheckpoint

# Project
import sys
sys.path.append('../common')
from data_utils import type_1_ids, type_2_ids, type_3_ids, test_ids
from data_utils import get_filename
from training_utils import get_trainval_id_type_lists, get_test_id_type_list, data_iterator
from metrics import logloss_mc
from resnet import get_resnet50


print("\n=========================")
print("Training dataset: ")
print("- type 1: ", len(type_1_ids))
print("- type 2: ", len(type_2_ids))
print("- type 3: ", len(type_3_ids))

print("Test dataset: ")
print("- ", len(test_ids))
print("=========================\n")


def train(model, train_id_type_list, val_id_type_list, batch_size=16, nb_epochs=10, image_size=(224, 224)):
    samples_per_epoch = 512
    nb_val_samples = 64

    if not os.path.exists('weights'):
        os.mkdir('weights')

    weights_filename = os.path.join("weights", "resnet50_simple.h5")
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True)

    print("Training parameters: ", batch_size, nb_epochs, samples_per_epoch, nb_val_samples)
    
    train_iter = data_iterator(train_id_type_list, batch_size=batch_size, image_size=image_size)
    val_iter = data_iterator(val_id_type_list, batch_size=batch_size, image_size=image_size)
    
    history = model.fit_generator(
        train_iter,
        steps_per_epoch=samples_per_epoch, 
        epochs=nb_epochs,
        validation_data=val_iter,
        validation_steps=nb_val_samples,
        callbacks=[model_checkpoint],
        verbose=2,
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
    print("\n- Get train/val lists ...")
    train_id_type_list, val_id_type_list = get_trainval_id_type_lists()
    print("\n- Get ResNet model ...")
    resnet = get_resnet50()
    print("\n- Start training ...")
    train(resnet, train_id_type_list, val_id_type_list)
    print("\n- Start validation ...")
    validate(resnet, val_id_type_list)
    print("\n- Start predictions and write submission ...")
    predict(resnet, info='resnet50_no_additional')
    





