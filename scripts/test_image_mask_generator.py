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
from training_utils import get_trainval_id_type_lists2, segmentation_xy_provider_RAM_cache as xy_provider, random_rgb_to_green

from preprocessing.image.generators import ImageMaskGenerator

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description="test_image_mask_generator.py")    
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size")
    parser.add_argument('--seed', type=int, default=2017, help="Numpy random seed")
           
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("\n {} - Get train/val lists ...".format(datetime.datetime.now()))        
    sloth_annotations_filename = os.path.join(RESOURCES_PATH, 'cervix_os.json')
    annotations = get_annotations(sloth_annotations_filename)
    train_id_type_list, val_id_type_list = get_trainval_id_type_lists2(annotations=annotations, val_split=0.25)
    print "Total : %s, Train : %s, Val : %s" % (len(annotations), len(train_id_type_list), len(val_id_type_list))
        
    batch_size = args.batch_size              
    
    train_gen = ImageMaskGenerator(pipeline=('random_transform', random_rgb_to_green, 'standardize'),
                                   featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=90., 
                                   width_shift_range=0.15, height_shift_range=0.15,
                                   shear_range=3.14/6.0,
                                   zoom_range=0.25,
                                   channel_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True)
    
    save_prefix='unet_os_cervix_detector'
    train_gen.fit(xy_provider(train_id_type_list, test_mode=True, image_size=(224, 224)),
                  len(train_id_type_list), 
                  augment=True, 
                  save_to_dir=GENERATED_DATA,
                  save_prefix=save_prefix,
                  batch_size=batch_size,
                  verbose=0)
    
    counter = 1024
    for x, y in train_gen.flow(xy_provider(train_id_type_list, image_size=(224, 224), verbose=1), 
                               len(train_id_type_list),
                               batch_size=batch_size):
        print "--", counter, x.shape, y.shape   
        counter -= 1
        if counter == 0:
            break
    
    
    print("\n {} - Scripted finished".format(datetime.datetime.now()))





