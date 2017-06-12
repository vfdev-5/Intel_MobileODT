import os
import pickle


import numpy as np
import cv2

from image_utils import get_image_data, get_os_image, get_cervix_image, scale_percentile
from data_utils import type_to_index


def image_provider(image_id_type_list, image_size=(224, 224), verbose=0, channels_first=True):
    for i, (image_id, image_type) in enumerate(image_id_type_list):
        if verbose > 0:
            print("Image id/type:", image_id, image_type, "| counter=", i)

        img = get_image_data(image_id, image_type)
        if img.dtype.kind is not 'u':
            if verbose > 0:
                print("Image is corrupted. Id/Type:", image_id, image_type)
            continue
        img = cv2.resize(img, dsize=image_size[::-1])
        if channels_first:
            img = img.transpose([2, 0, 1])
        img = img.astype(np.float32) / 255.0

        yield img, img, (image_id, image_type)


def image_mask_provider(image_id_type_list,
                        label_type,
                        image_size=(224, 224),
                        channels_first=True,
                        test_mode=False,
                        save_to_dir=None,
                        verbose=0):
    image_id_type_list = list(image_id_type_list)
    while True:
        np.random.shuffle(image_id_type_list)
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", i)

            output_filename = 'preproc_img_label_'+ image_id + "_" + image_type + '.npz'
            output_filename = os.path.join(save_to_dir, output_filename) if save_to_dir is not None else output_filename
            if os.path.exists(output_filename):
                print("-- Load preprocessed image")
                data = np.load(output_filename)
                img = data['image']
                label = data['label']
            else:
                img = get_image_data(image_id, image_type)
                if img.dtype.kind is not 'u':
                    if verbose > 0:
                        print("Image is corrupted. Id/Type:", image_id, image_type)
                    continue
                img = cv2.resize(img, dsize=image_size[::-1])
                if channels_first:
                    img = img.transpose([2, 0, 1])
                img = img.astype(np.float32) / 255.0

                label = get_image_data(image_id + "_" + image_type, label_type)
                label = cv2.resize(label, dsize=image_size[::-1])
                if channels_first:
                    label = label.transpose([2, 0, 1])

                if save_to_dir is not None:
                    if not os.path.exists(save_to_dir):
                        os.makedirs(save_to_dir)
                    np.savez(output_filename, image=img, label=label)

            if test_mode:
                yield img, label, (image_id, image_type)
            else:
                yield img, label

        if test_mode:
            return


class DataCache(object):
    """
    Queue storage of any data to avoid reloading
    """
    def __init__(self, n_samples):
        """
        :param n_samples: max number of data items to store in RAM
        """
        self.n_samples = n_samples
        self.cache = {}
        self.ids_queue = []

    def put(self, data_id, data):

        if 0 < self.n_samples < len(self.cache):
            key_to_remove = self.ids_queue.pop(0)
            self.cache.pop(key_to_remove)

        self.cache[data_id] = data
        if data_id in self.ids_queue:
            self.ids_queue.remove(data_id)
        self.ids_queue.append(data_id)

    def get(self, data_id):
        return self.cache[data_id]

    def remove(self, data_id):
        self.ids_queue.remove(data_id)
        self.cache.pop(data_id)

    def __contains__(self, key):
        return key in self.cache and key in self.ids_queue


def save_data_cache(cache, filename):
    assert not os.path.exists(filename), "Output file already exists"
    with open(filename, 'wb') as f:
        pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)


def load_data_cache(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def cached_image_mask_provider(image_id_type_list,
                               mask_type,
                               image_size=(224, 224),
                               channels_first=True,
                               test_mode=False,
                               cache=None,
                               verbose=0):

    if cache is None:
        img = get_image_data(*image_id_type_list[0])
        n_channels = img.shape[2]
        # 2(image/mask) * N * 224 * 224 * 3 (rgb) * 4 (float32) = 2 * N * 602,112 bytes
        # Let us assume that cache can load 650 Mb
        n_img = int(650.0 * 1e6 / (2.0 * image_size[0] * image_size[1] * n_channels * 4))
        n_img = min(n_img, len(image_id_type_list))
        cache = DataCache(n_img)
        if verbose > 0:
            print("Initialize cache : %i" % cache.n_samples)

    counter = 0
    image_id_type_list = list(image_id_type_list)
    while True:
        np.random.shuffle(image_id_type_list)
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", i)

            key = image_id + '_' + image_type
            if key in cache:
                if verbose > 0:
                    print("-- Load from RAM")
                img, mask = cache.get(key)
            else:
                if verbose > 0:
                    print("-- Load from disk")
                img = get_image_data(image_id, image_type)
                if img.dtype.kind is not 'u':
                    if verbose > 0:
                        print("Image is corrupted. Id/Type:", image_id, image_type)
                    continue
                img = cv2.resize(img, dsize=image_size[::-1])
                if channels_first:
                    img = img.transpose([2, 0, 1])
                img = img.astype(np.float32) / 255.0
                mask = get_image_data(image_id + "_" + image_type, mask_type)
                mask = cv2.resize(mask, dsize=image_size[::-1])
                if channels_first:
                    mask = mask.transpose([2, 0, 1])

                # fill the cache only at first time:
                if counter == 0:
                    cache.put(key, (img, mask))
            if test_mode:
                yield img, mask, (image_id, image_type)
            else:
                yield img, mask

        if test_mode:
            return
        counter += 1


# def cached_image_provider(image_id_type_list,
#                           image_size=(224, 224),
#                           channels_first=True,
#                           option="",
#                           cache=None,
#                           verbose=0):
#     if cache is None:
#         cache = DataCache(n_samples=500)
#
#     for i, (image_id, image_type) in enumerate(image_id_type_list):
#         if verbose > 0:
#             print("Image id/type:", image_id, image_type, "| counter=", i)
#
#         key = (image_id, image_type, option)
#         if key in cache:
#             img, _ = cache.get(key)
#
#             if channels_first:
#                 if img.shape[1:] != image_size[::-1]:
#                     img = img.transpose([1, 2, 0])
#                     img = cv2.resize(img, dsize=image_size[::-1])
#                     img = img.transpose([2, 0, 1])
#             else:
#                 if img.shape[:2] != image_size[::-1]:
#                     img = cv2.resize(img, dsize=image_size[::-1])
#
#         else:
#             if option == 'cervix':
#                 img = get_cervix_image(image_id, image_type)
#             elif option == 'os':
#                 img = get_os_image(image_id, image_type)
#             else:
#                 img = get_image_data(image_id, image_type)
#
#             if img.dtype.kind is not 'u':
#                 if verbose > 0:
#                     print("Image is corrupted. Id/Type:", image_id, image_type)
#                 continue
#             img = cv2.resize(img, dsize=image_size[::-1])
#             if channels_first:
#                 img = img.transpose([2, 0, 1])
#             img = img.astype(np.float32) / 255.0
#             cache.put(key, (img, None))
#
#         yield img, None, (image_id, image_type)


def cached_image_label_provider(image_id_type_list,
                                image_size,
                                option=None,  # 'cervix', 'os' or 'cervix/os'
                                channels_first=True,
                                test_mode=False,
                                cache=None,
                                seed=None,
                                with_labels=True,
                                verbose=0):

    if seed is not None:
        np.random.seed(seed)

    if cache is None:
        img = get_image_data(*image_id_type_list[0])
        image_height, image_width, n_channels = img.shape
        # Let us assume that cache can load 650 Mb
        n_img = int(650.0 * 1e6 / (1.0 * image_width * image_height * n_channels * 4.0))
        n_img = min(n_img, len(image_id_type_list))
        cache = DataCache(n_img)
        if verbose > 0:
            print("Initialize cache : %i" % cache.n_samples)

    counter = 0
    image_id_type_list = list(image_id_type_list)
    while True:
        np.random.shuffle(image_id_type_list)
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", i)

            key = (image_id, image_type, option)
            if key in cache:
                if verbose > 0:
                    print("-- Load from RAM")
                img, label = cache.get(key)

                if channels_first:
                    if img.shape[1:] != image_size[::-1]:
                        img = img.transpose([1, 2, 0])
                        img = cv2.resize(img, dsize=image_size[::-1])
                        img = img.transpose([2, 0, 1])
                else:
                    if img.shape[:2] != image_size[::-1]:
                        img = cv2.resize(img, dsize=image_size[::-1])
            else:
                if verbose > 0:
                    print("-- Load from disk")

                if option == 'cervix':
                    img = get_cervix_image(image_id, image_type)
                elif option == 'os':
                    img = get_os_image(image_id, image_type)
                else:
                    img = get_image_data(image_id, image_type)

                if img.shape[:2] != image_size:
                    img = cv2.resize(img, dsize=image_size)
                if channels_first:
                    img = img.transpose([2, 0, 1])

                img = img.astype(np.float32) / 255.0

                if with_labels:
                    label = np.array([0, 0, 0], dtype=np.uint8)
                    label[type_to_index[image_type]] = 1
                else:
                    label = None
                # fill the cache only at first time:
                if counter == 0:
                    cache.put(key, (img, label))
            if test_mode:
                yield img, label, (image_id, image_type)
            else:
                yield img, label

        if test_mode:
            return
        counter += 1
        
        
def cached_image_value_provider(image_id_type_list,
                                image_size,
                                option=None,  # 'cervix', 'os' or 'cervix/os'
                                channels_first=True,
                                test_mode=False,
                                cache=None,
                                seed=None,
                                with_labels=True,
                                verbose=0):

    if seed is not None:
        np.random.seed(seed)

    if cache is None:
        img = get_image_data(*image_id_type_list[0])
        image_height, image_width, n_channels = img.shape
        # Let us assume that cache can load 650 Mb
        n_img = int(650.0 * 1e6 / (1.0 * image_width * image_height * n_channels * 4.0))
        n_img = min(n_img, len(image_id_type_list))
        cache = DataCache(n_img)
        if verbose > 0:
            print("Initialize cache : %i" % cache.n_samples)

    counter = 0
    image_id_type_list = list(image_id_type_list)
    while True:
        np.random.shuffle(image_id_type_list)
        for i, (image_id, image_type) in enumerate(image_id_type_list):
            if verbose > 0:
                print("Image id/type:", image_id, image_type, "| counter=", i)

            key = (image_id, image_type, option)
            if key in cache:
                if verbose > 0:
                    print("-- Load from RAM")
                img, label = cache.get(key)

                if channels_first:
                    if img.shape[1:] != image_size[::-1]:
                        img = img.transpose([1, 2, 0])
                        img = cv2.resize(img, dsize=image_size[::-1])
                        img = img.transpose([2, 0, 1])
                else:
                    if img.shape[:2] != image_size[::-1]:
                        img = cv2.resize(img, dsize=image_size[::-1])
            else:
                if verbose > 0:
                    print("-- Load from disk")

                if option == 'cervix':
                    img = get_cervix_image(image_id, image_type)
                elif option == 'os':
                    img = get_os_image(image_id, image_type)
                else:
                    img = get_image_data(image_id, image_type)

                if img.shape[:2] != image_size:
                    img = cv2.resize(img, dsize=image_size)
                if channels_first:
                    img = img.transpose([2, 0, 1])

                img = img.astype(np.float32) / 255.0

                if with_labels:                    
                    # type_to_index[image_type] = [0, 1, 2]
                    label = np.array([type_to_index[image_type]*(-500.0) + np.random.randint(100) + 450.0,])
                else:
                    label = None
                # fill the cache only at first time:
                if counter == 0:
                    cache.put(key, (img, label))
            if test_mode:
                yield img, label, (image_id, image_type)
            else:
                yield img, label

        if test_mode:
            return
        counter += 1
