
import numpy as np
import cv2


def sieve(image, size=None, compactness=None, use_convex_hull=False):
    """
    Filter removes small objects of 'size' from binary image
    Input image should be a single band image of type np.uint8

    Idea : use Opencv findContours
    """
    assert image.dtype == np.uint8, "Input should be a Numpy array of type np.uint8"
    assert size is not None or compactness is not None, "Either size or compactness should be defined"

    if size is not None:
        sq_limit = size** 2
        lin_limit = size * 4

    out_image = image.copy()
    image, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if hierarchy is not None and len(hierarchy) > 0:
        hierarchy = hierarchy[0]
        index = 0
        while index >= 0:
            contour = contours[index]

            if use_convex_hull:
                contour = cv2.convexHull(contour)
            p = cv2.arcLength(contour, True)
            s = cv2.contourArea(contour)
            r = cv2.boundingRect(contour)
            if size is not None:
                if s <= sq_limit and p <= lin_limit:
                    out_image[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = 0
            if compactness is not None:
                if np.sign(compactness) * 4.0 * np.pi * s / p ** 2 > np.abs(compactness):
                    out_image[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = 0
            # choose next contour of the same hierarchy
            index = hierarchy[index][0]

    return out_image


def treshold(img, t=0.75):
    out = img.copy()
    out[out >= t] = 1.0
    out[out < t] = 0.0
    return out.astype(np.uint8)


def keep_one_object(image, largest=True, postproc_contour_f=None):
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) == 0:
        return image

    out_contour = cnts[0]
    area = cv2.contourArea(out_contour)
    op = lambda x, y: x > y if largest else x < y
    for c in cnts[1:]:
        s = cv2.contourArea(c)
        if op(s, area):
            area = s
            out_contour = c

    if postproc_contour_f is not None:
        out_contour = postproc_contour_f(out_contour)

    out_image = np.zeros_like(image)
    out_image = cv2.drawContours(out_image, (out_contour,), 0, (255), thickness=cv2.FILLED)

    return out_image


def cervix_postproc(img):
    """
    - If touches two opposite boundaries -> fill whole image
    - Only one largest os on the image
    - Round form -> morpho close
    - No holes -> morpho close
    - Enlarge zone -> convex hull
    """
    out = img
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                           iterations=1)
    # detect cervix at opposite boundaries
    h, w = out.shape
    margin = 15
    b1 = np.sum(out[:margin, :]) > w / 5 and np.sum(out[-margin + 1:, :]) > w / 5
    b2 = np.sum(out[:, :margin]) > h / 5 and np.sum(out[:, -margin + 1:]) > h / 5

    if b1 or b2:
        out[margin / 2:-margin / 2 + 1, margin / 2:-margin / 2 + 1] = 1
        return out

    out = keep_one_object(out, largest=True, postproc_contour_f=cv2.convexHull)

    return out


def os_postproc(img):
    """
    - Only one largest os on the image
    - No holes
    """
    out = img
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                           iterations=2)

    out = keep_one_object(out, largest=True)
    return out


def os_cervix_postproc(mask):
    mask = treshold(mask)
    mask[:, :, 0] = os_postproc(mask[:, :, 0])  # Os
    mask[:, :, 1] = cervix_postproc(mask[:, :, 1])  # Cervix
    mask[:, :, 0] *= mask[:, :, 1]
    return mask


def os_cervix_postproc_batch(y_pred):
    y_pred = treshold(y_pred)
    for i in range(y_pred.shape[0]):
        y_pred[i, 0, :, :] = os_postproc(y_pred[i, 0, :, :])  # Os
        y_pred[i, 1, :, :] = cervix_postproc(y_pred[i, 1, :, :])  # Cervix
        y_pred[i, 0, :, :] *= y_pred[i, 1, :, :]

    return y_pred


def os_cervix_merge_masks(os_cervix_mask_1, os_cervix_mask_2, a=0.5, b=0.5):
    os_cervix_1_sum = np.sum(os_cervix_mask_1 > 0.5, axis=(0, 1))
    os_cervix_2_sum = np.sum(os_cervix_mask_2 > 0.5, axis=(0, 1))
    os_cervix_indices = np.abs(os_cervix_1_sum - os_cervix_2_sum) * 1.0 / (os_cervix_1_sum + os_cervix_2_sum + 1e-7)    
    
    final_os_cervix_mask = np.zeros_like(os_cervix_mask_1)    
    for i in range(2):
        if os_cervix_indices[i] < 0.5:
            final_os_cervix_mask[:,:,i] = a * os_cervix_mask_1[:,:,i] + b * os_cervix_mask_2[:,:,i]
        else:
            if os_cervix_1_sum[i] > os_cervix_2_sum[i]:
                final_os_cervix_mask[:,:,i] = os_cervix_mask_1[:,:,i]
            else:
                final_os_cervix_mask[:,:,i] = os_cervix_mask_2[:,:,i]
    return final_os_cervix_mask


def get_bbox(mask):
    img_proj_x = np.sum(mask, axis=0)
    img_proj_y = np.sum(mask, axis=1)
    indices = np.where(img_proj_x > 0)
    xmin = indices[0][0]
    xmax = indices[0][-1]
    indices = np.where(img_proj_y > 0)
    ymin = indices[0][0]
    ymax = indices[0][-1]
    return [xmin, ymin, xmax, ymax]


def crop_to_mask(img, mask):
    img = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_proj_x = np.sum(gray, axis=0)
    img_proj_y = np.sum(gray, axis=1)
    indices = np.where(img_proj_x > 0)
    xmin = indices[0][0]
    xmax = indices[0][-1]
    indices = np.where(img_proj_y > 0)
    ymin = indices[0][0]
    ymax = indices[0][-1]
    return img[ymin:ymax, xmin:xmax, :]
