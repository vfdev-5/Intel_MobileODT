import os
import sys
import datetime
import argparse
from glob import glob
    
import numpy as np

# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)


import platform

if 'c001' in platform.node():
    from colfax_configuration import setup_keras_122
    setup_keras_122()


# Project
from data_utils import RESOURCES_PATH, GENERATED_DATA, get_annotations
from training_utils import get_trainval_id_type_lists2
from image_utils import get_image_data
from metrics import logloss_mc
from unet_keras_v1 import get_unet

from test_utils import get_test_id_type_list2
from training_utils import segmentation_train as train, segmentation_validate as validate
from test_utils import segmentation_predict as predict


if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description="unet_cervix_os_detection_with_keras.py")
    
    parser.add_argument('--bypass-train', action='store_true', help="Train model")
    parser.add_argument('--load-best-weights', action='store_true', help="Load pretrained best weights")
    parser.add_argument('--seed', type=int, default=2017, help="Numpy random seed")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size")
    parser.add_argument('--nb-epochs', type=int, default=50, help="Nb epochs")
           
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("\n {} - Get train/val lists ...".format(datetime.datetime.now()))        
    sloth_annotations_filename = os.path.join(RESOURCES_PATH, 'cervix_os.json')
    annotations = get_annotations(sloth_annotations_filename)
    train_id_type_list, val_id_type_list = get_trainval_id_type_lists2(annotations=annotations, val_split=0.25)
    print "Total : %s, Train : %s, Val : %s" % (len(annotations), len(train_id_type_list), len(val_id_type_list))
        
    print("\n {} - Get U-Net model ...".format(datetime.datetime.now()))
    unet = get_unet(input_shape=(3, 224, 224), n_classes=2)
    
    save_prefix='unet_os_cervix_detector' # + datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    if args.load_best_weights:        
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

    batch_size = args.batch_size              
    if not args.bypass_train:
        nb_epochs = args.nb_epochs
        print("\n {} - Start training ...".format(datetime.datetime.now()))
        train(unet, train_id_type_list, val_id_type_list, 
              save_prefix=save_prefix, nb_epochs=nb_epochs, batch_size=batch_size, verbose=2)
    
    print("\n {} - Start validation ...".format(datetime.datetime.now()))    
    validate(unet, val_id_type_list, batch_size=batch_size)
    
    print("\n {} - Start predictions and write detections".format(datetime.datetime.now()))
    test_id_type_list = get_test_id_type_list2(annotations)    
    predict(unet, test_id_type_list, batch_size=batch_size)
    print("\n {} - Scripted finished".format(datetime.datetime.now()))





