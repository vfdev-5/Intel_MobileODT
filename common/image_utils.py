
import os
import cv2
from PIL import Image
import numpy as np

# Project
from data_utils import get_filename


def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    if image_type == 'label':
        return np.load(get_filename(image_id, image_type))['arr_0']
    return _get_image_data_pil(image_id, image_type)


def imwrite(img, image_id, image_type):
    output_filename = get_filename(image_id, image_type)
    if image_type == 'label':
        np.savez(output_filename, img)
    else:
        pil_image = Image.fromarray(img)
        pil_image.save(output_filename)


def _get_image_data_opencv(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _get_image_data_pil(image_id, image_type, return_exif_md=False):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    try:
        img_pil = Image.open(fname)
    except Exception as e:
        assert False, "Failed to read image : %s, %s. Error message: %s" % (image_id, image_type, e)

    img = np.asarray(img_pil)
    assert isinstance(img, np.ndarray), "Open image is not an ndarray. Image id/type : %s, %s" % (image_id, image_type)
    if not return_exif_md:
        return img
    else:
        return img, img_pil._getexif()


def label_to_index(label):
    try:
        index = ['os', 'cervix'].index(label)
    except:
        raise Exception("Label '%s' is unknown" % label)
    return index


def generate_label_gray_images(annotations):
    n = len(annotations)
    for i, annotation in enumerate(annotations):

        image_name = annotation['filename']
        image_id = os.path.basename(image_name)[:-4]
        image_type = os.path.split(os.path.dirname(image_name))[1]
        src_image = get_image_data(image_id, image_type)
        
        print('--', image_id, image_type, i, '/', n)
        ll = len(annotation['annotations'])
        label_image = np.zeros(src_image.shape[:2], dtype=np.uint8)
        for label in annotation['annotations']:
            
            assert label['type'] == u'rect', "Type '%s' is not supported" % label['type']
            index = label_to_index(label['class']) + 1
            pt1 = (int(label['x']), int(label['y']))
            pt2 = (pt1[0] + int(label['width']), pt1[1] + int(label['height']))
            mask = label_image[pt1[1]:pt2[1], pt1[0]:pt2[0]] == 0            
            label_image[pt1[1]:pt2[1], pt1[0]:pt2[0]] = index * mask + label_image[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        
        imwrite(label_image, image_id + '_' + image_type, 'label_gray')  
        

def generate_label_images(annotations):
    n = len(annotations)

    def _clamp(x, dim):
        return min(max(x, 0), dim - 1)

    for i, annotation in enumerate(annotations):

        image_name = annotation['filename']
        image_id = os.path.basename(image_name)[:-4]
        image_type = os.path.split(os.path.dirname(image_name))[1]
        src_image = get_image_data(image_id, image_type)
        
        print('--', image_id, image_type, i, '/', n)
        ll = len(annotation['annotations'])
        label_image = np.zeros(src_image.shape[:2] + (ll,), dtype=np.uint8)
        h, w, _ = label_image.shape
        for label in annotation['annotations']:
            assert label['type'] == u'rect', "Type '%s' is not supported" % label['type']
            index = label_to_index(label['class'])
            pt1 = (_clamp(int(label['x']), w), _clamp(int(label['y']), h))
            pt2 = (_clamp(pt1[0] + int(label['width']), w), _clamp(pt1[1] + int(label['height']), h))
            label_image[pt1[1]:pt2[1], pt1[0]:pt2[0], index] = 1
        
        imwrite(label_image, image_id + '_' + image_type, 'label')    