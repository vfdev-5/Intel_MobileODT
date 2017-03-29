import os
import sys
import datetime
from glob import glob
    
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 

# On Colfax :
if os.path.exists("/home/u2459/keras-1.2.2"):
    keras_lib_path = "/home/u2459/keras-1.2.2/build/lib"
    if not keras_lib_path in sys.path:
        sys.path.insert(0, "/home/u2459/keras-1.2.2/build/lib")
    from keras import __version__
    print "Keras version: ", __version__
    import theano
    print "mkl_available: ", theano.sandbox.mkl.mkl_available()


# Project
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common'))
from data_utils import RESOURCES_PATH, GENERATED_DATA, get_annotations
from training_utils import get_trainval_id_type_lists2
from image_utils import get_image_data
from metrics import logloss_mc
from unet_keras_v1 import get_unet

from test_utils import get_test_id_type_list2
from training_utils import segmentation_train as train, segmentation_validate as validate
from test_utils import segmentation_predict as predict

np.random.seed(2017)       

if __name__ == "__main__":

    import platform
    
    print("\n {} - Get train/val lists ...".format(datetime.datetime.now()))        
    sloth_annotations_filename = os.path.join(RESOURCES_PATH, 'cervix_os.json')
    annotations = get_annotations(sloth_annotations_filename)
    train_id_type_list, val_id_type_list = get_trainval_id_type_lists2(annotations=annotations, val_split=0.25)
    print "Total : %s, Train : %s, Val : %s" % (len(annotations), len(train_id_type_list), len(val_id_type_list))
        
    print("\n {} - Get U-Net model ...".format(datetime.datetime.now()))
    unet = get_unet(input_shape=(3, 224, 224), n_classes=2)
    
    save_prefix='unet_os_cervix_detector' # + datetime.now().strftime("%Y-%m-%d-%H-%M")
    weights_files = glob("weights/%s*.h5" % save_prefix)
    best_val_loss = 1e5
    best_weights_filename = ""
    for f in weights_files:
        index = os.path.basename(f).index('-')
        loss = float(os.path.basename(f)[index+1:-4])
        if best_val_loss > loss:
            best_val_loss = loss
            best_weights_filename = f
    
    if len(best_weights_filename) > 0:
        # load weights to the model
        print("Load found weights: ", best_weights_filename)
        unet.load_weights(best_weights_filename)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        nb_epochs = 50
        batch_size = 4   
        print("\n {} - Start training ...".format(datetime.datetime.now()))
        train(unet, train_id_type_list, val_id_type_list, nb_epochs=nb_epochs, batch_size=batch_size)
    
    print("\n {} - Start validation ...".format(datetime.datetime.now()))
    batch_size = 4 
    validate(unet, val_id_type_list, batch_size=batch_size)
    
    print("\n {} - Start predictions and write detections".format(datetime.datetime.now()))
    test_id_type_list = get_test_id_type_list2(annotations)    
    batch_size = 4 
    predict(unet, test_id_type_list, batch_size=batch_size)
    print("\n {} - Scripted finished".format(datetime.datetime.now()))





