
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
        pass
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
