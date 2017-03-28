import os
import datetime

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 

from keras.callbacks import ModelCheckpoint

# Project
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common'))
from data_utils import type_1_ids, type_2_ids, type_3_ids, test_ids
from data_utils import RESOURCES_PATH, GENERATED_DATA, get_annotations
from training_utils import get_trainval_id_type_lists2, get_test_id_type_list2
from image_utils import get_image_data
from metrics import logloss_mc
from unet_keras_v1 import get_unet

# Local keras-contrib:
from preprocessing.image.generators import ImageMaskGenerator
from preprocessing.image.iterators import XYIterator


np.random.seed(2017)


def xy_provider(image_id_type_list, 
                image_size=(224, 224), 
                test_mode=False,
                verbose=0):        
    while True:
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", counter)

            img = get_image_data(image_id, image_type)
            if img.dtype.kind is not 'u':
                if verbose > 0:
                    print("Image is corrupted. Id/Type:", image_id, image_type)
                continue
            img = cv2.resize(img, dsize=image_size[::-1])
            img = img.transpose([2, 0, 1])
            img = img.astype(np.float32) / 255.0

            label = get_image_data(image_id + "_" + image_type, "label")
            label = cv2.resize(label, dsize=image_size[::-1])
            label = label.transpose([2, 0, 1])

            yield img, label
        if test_mode:
            return
        

def train(model, train_id_type_list, val_id_type_list, batch_size=16, nb_epochs=10, image_size=(224, 224)):
    
    samples_per_epoch = (512 // batch_size) * batch_size
    nb_val_samples = (128 // batch_size) * batch_size
    #samples_per_epoch = (2048 // batch_size) * batch_size
    #nb_val_samples = (1024 // batch_size) * batch_size

    if not os.path.exists('weights'):
        os.mkdir('weights')

    weights_filename = os.path.join("weights", "unet_os_cervix_detector_{epoch:02d}-{val_loss:.4f}.h5")
    model_checkpoint = ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True)

    print("Training parameters: ", batch_size, nb_epochs, samples_per_epoch, nb_val_samples)
    
    train_gen = ImageMaskGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=90., 
                                   width_shift_range=0.15, height_shift_range=0.15,
                                   shear_range=3.14/6.0,
                                   zoom_range=0.25,
                                   channel_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True)
    val_gen = ImageMaskGenerator(rotation_range=90., 
                                 horizontal_flip=True,
                                 vertical_flip=True)
    
    train_gen.fit(xy_provider(train_id_type_list, test_mode=True),
                  len(train_id_type_list), 
                  augment=True, 
                  save_to_dir=GENERATED_DATA,
                  save_prefix='os_cervix',
                  batch_size=4,
                  verbose=1)
   
    history = model.fit_generator(
        train_gen.flow(xy_provider(train_id_type_list), 
                       len(train_id_type_list),
                       batch_size=batch_size),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epochs,
        validation_data=val_gen.flow(xy_provider(val_id_type_list), 
                       len(val_id_type_list),
                       batch_size=batch_size),
        nb_val_samples=nb_val_samples,
        callbacks=[model_checkpoint],
        verbose=2,
    )

    return history


def validate(model, val_id_type_list, batch_size=16, image_size=(224, 224)):
      
    val_iter = XYIterator(xy_provider(val_id_type_list, test_mode=True), 
                          len(val_id_type_list), 
                          None, # image generator
                          batch_size=batch_size,
                          data_format='channels_first')
    
    total_loss = 0.0
    total_counter = 0 
    for x, y_true in val_iter:           
        s = y_true.shape[0]
        total_counter += s
        y_pred = model.predict(x)
        loss = logloss_mc(y_true, y_pred)
        total_loss += s * loss
        print("--", total_counter, "batch loss : ", loss)

    if total_counter == 0:
        total_counter += 1

    total_loss *= 1.0 / total_counter   
    print("Total loss : ", total_loss)
    
    
def predict(model, test_id_type_list, batch_size=16, image_size=(224, 224), info=''):

    
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
    
    nb_epochs = 10
    batch_size = 8    

    print("\n {} - Get train/val lists ...".format(datetime.datetime.now()))        
    sloth_annotations_filename = os.path.join(RESOURCES_PATH, 'cervix_os.json')
    annotations = get_annotations(sloth_annotations_filename)

    train_id_type_list, val_id_type_list = get_trainval_id_type_lists2(annotations=annotations, val_split=0.25)

    print "Total : %s, Train : %s, Val : %s" % (len(annotations), len(train_id_type_list), len(val_id_type_list))
        
    print("\n {} - Get U-Net model ...".format(datetime.datetime.now()))
    unet = get_unet(input_shape=(3, 224, 224), n_classes=2)
    print("\n {} - Start training ...".format(datetime.datetime.now()))
    train(unet, train_id_type_list, val_id_type_list, nb_epochs=nb_epochs, batch_size=batch_size)
    #print("\n {} - Start validation ...".format(datetime.datetime.now()))
    #validate(unet, val_id_type_list, batch_size=batch_size)
    #test_id_type_list = get_test_id_type_list2(annotations)
    #print("\n {} - Start predictions and write detections".format(datetime.datetime.now()))
    #predict(unet, info='unet_no_additional', batch_size=batch_size)
    #print("\n {} - Scripted finished".format(datetime.datetime.now()))





