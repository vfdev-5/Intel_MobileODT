import os
import datetime

import numpy as np
import cv2
import pandas as pd

# Project
from data_utils import test_ids, type_to_index, type_1_ids, type_2_ids, type_3_ids
from image_utils import imwrite, get_image_data

# Local keras-contrib:
from preprocessing.image.iterators import ImageMaskIterator


def get_test_id_type_list():
    return [(image_id, 'Test') for image_id in test_ids]


def get_test_id_type_list2(annotations):
    trainval_id_type_list = []
    for annotation in annotations:
        image_name = annotation['filename']
        image_id = os.path.basename(image_name)[:-4]
        image_type = os.path.split(os.path.dirname(image_name))[1]
        trainval_id_type_list.append((image_id, image_type))

    test_id_type_list = [(image_id, 'Test') for image_id in test_ids]
    type_ids=(type_1_ids, type_2_ids, type_3_ids)
    image_types = ["Type_1", "Type_2", "Type_3"]

    for image_ids, image_type in zip(type_ids, image_types):    
        for image_id in image_ids:
            if (image_id, image_type) not in trainval_id_type_list:
                test_id_type_list.append((image_id, image_type))
    return test_id_type_list


### Segmentation

def segmentation_xy_provider(image_id_type_list, image_size=(224, 224), verbose=0):        
    
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

        yield img, img, (image_id, image_type)
                
        
def segmentation_predict(model, test_id_type_list, batch_size=16, image_size=(224, 224)):
    
    test_iter = ImageMaskIterator(segmentation_xy_provider(test_id_type_list, image_size=image_size), 
                                  len(test_id_type_list), 
                                  None, # image generator
                                  batch_size=batch_size,
                                  data_format='channels_first')
    
    total_counter = 0
    for x, _, info in test_iter:            
        y_pred = model.predict(x)    
        s = y_pred.shape[0]            
        for i in range(s):
            total_counter += 1
            print("--", total_counter, info[i])
            imwrite(y_pred[i, :, :, :], info[i][0] + '_' + info[i][1], 'pred_label')
