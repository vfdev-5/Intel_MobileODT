
import numpy as np

from keras import backend as K


def compute_heatmap(model, image, mask_size=50, step=10, n_top_classes=5, batch_size=16):
    """
    Method to compute heatmap using a classification model on preprocessed image. 
    Idea is to draw a black box on the input image, slide it and compute a prediction.
    
    :param model: compiled keras model ready to execute `predict` method.
    :param image: input preprocessed image, ndarray of shape (H, W, C) or (C, H, W).
    :param mask_size: size in pixels of the black mask on the original image.
    :param step: step in pixels between to consecutive black masks.
    :param n_top_classes: produce heatmap on originally predicted top k classes. If n_top_classes==None
    Returned heatmap contains all classes.
    :param batch_size: batch size used for predictions on images with black boxes.    
    
    :return: (heatmap, top_class_indices) ndarray of shape (n_top_classes, H, W). Heatmap values are positive and 
    represent the difference between original prediction and the predicition when a part of image is masked.
    """
    assert isinstance(image, np.ndarray) and len(image.shape) == 3, \
        "Input image should be a ndarray of shape (H, W, C) or (C, H, W)"
    
    if hasattr(model, 'outputs'):
        outputs = model.outputs
    elif hasattr(model, 'output'):
        outputs = model.output
    else:
        raise Exception("Model should have attribute 'outputs' or 'output'")
     
    if isinstance(outputs, list):
        assert len(outputs) == 1, "Model should have a single output"          
        outputs = outputs[0]
     
    assert len(outputs._keras_shape) == 2, "Model should perform classification task"
    n_classes = outputs._keras_shape[1]
    if n_top_classes is None:
        n_top_classes = n_classes
        
    channel_axis = np.argmin(image.shape)
    assert channel_axis == 0 or channel_axis == 2, "Input image shape should be (H, W, C) or (C, H, W)"
            
    if not hasattr(K, 'image_data_format') and hasattr(K, 'image_dim_ordering'):
        data_format = "channels_last" if K.image_dim_ordering() == "tf" else "channels_first"                       
    else:
        data_format = K.image_data_format()
        
    # convert input image to channels_first format
    if channel_axis == 2:
        _image = image.transpose([2, 0, 1])
    else:
        _image = image
                                                    
    # define batch with 'channels_format'
    batch = np.zeros((batch_size,) + _image.shape, dtype=np.float32)
    _to_data_format = lambda x : x
    if data_format == 'channels_last':
        batch = batch.transpose([0, 2, 3, 1])
        _to_data_format = lambda image : image.transpose([1, 2, 0])

    # compute predictions on the original image:        
    batch[0, :, :, :] = _to_data_format(_image[:, :, :])
    y_pred_0 = model.predict_on_batch(batch[:1,:,:,:]) 
    print("y_pred_0=", y_pred_0)
    heatmap = np.zeros((n_top_classes,) + _image.shape[1:])
    top_class_indices = np.argsort(y_pred_0[0, :])[::-1][:n_top_classes]
    print("top_class_indices=", top_class_indices)
    #for i in range(n_top_classes):
    #    heatmap[i, :, :] = y_pred_0[0, top_class_indices[i]]
                  
    def _update_heatmap():
        for j, (_x, _y) in enumerate(batch_mask_xy):
            for i in range(n_top_classes):
                diff = y_pred_0[0, top_class_indices[i]] - y_pred[j, top_class_indices[i]]
                heatmap[i, _y:_y+mask_size, _x:_x+mask_size] = np.maximum(diff, heatmap[i, _y:_y+mask_size, _x:_x+mask_size])
 
        
    counter = 0
    batch_mask_xy = []
    for x in range(0, _image.shape[2], step):
        for y in range(0, _image.shape[1], step):
            image_copy = _image.copy()
            image_copy[:, y:y+mask_size, x:x+mask_size] = 0
            batch[counter, :, :, :] = _to_data_format(image_copy[:, :, :])
            batch_mask_xy.append((x, y))
            counter += 1
            if counter == batch_size:
                y_pred = model.predict_on_batch(batch)
                _update_heatmap()
                batch_mask_xy = []
                counter = 0
               
    if counter > 0:
        y_pred = model.predict_on_batch(batch[:counter, :, :, :])
        _update_heatmap()
       
    return heatmap, top_class_indices




def compute_heatmap_v0(model, image, resolution=0.9, top_k_classes=5, batch_size=4):
    """
    Method to compute heatmap using a classification model on preprocessed image. 
    Idea is to draw a black box on the input image, slide it and compute a prediction.
    
    :param model: compiled keras model ready to execute `predict` method.
    :param image: input preprocessed image, ndarray of shape (H, W, C) or (C, H, W).
    :param resolution: is a variable between 0 and 1 to compute black box size. 
        ```bbox_size = image.size * 2 ** ( -(resolution*7.0 + 1.0))```
    :param top_k_classes: produce heatmap on originally predicted top k classes. If top_k_classes==None
    Returned heatmap contains all classes.
    :param batch_size: batch size used for predictions on images with black boxes.    
    
    :return heatmap: ndarray of shape (top_k_classes, H, W)
    """
    assert 0 <= resolution <= 1, "Resolution should be between 0 and 1"
    assert isinstance(image, np.ndarray) and len(image.shape) == 3, \
        "Input image should be a ndarray of shape (H, W, C) or (C, H, W)"
    
    if hasattr(model, 'outputs'):
        outputs = model.outputs
    elif hasattr(model, 'output'):
        outputs = model.output
    else:
        raise Exception("Model should have attribute 'outputs' or 'output'")
     
    if isinstance(outputs, list):
        assert len(outputs) == 1, "Model should have a single output"          
        outputs = outputs[0]
     
    assert len(outputs._keras_shape) == 2, "Model should perform classification task"
    n_classes = outputs._keras_shape[1]
    if top_k_classes is None:
        top_k_classes = n_classes
        
    channel_axis = np.argmin(image.shape)
    assert channel_axis == 0 or channel_axis == 2, "Input image shape should be (H, W, C) or (C, H, W)"

    def _blackbox_coords_iterator(h, w, size):
        
        #def _compute_n(image_size, bbox_size, min_overlapping):
        #    return int(np.ceil(image_size * 1.0 / (bbox_size - min_overlapping)))
        
        def _compute_n(image_size, bbox_size):
            return int(np.ceil(image_size * 1.0 / bbox_size))
        
        #def _compute_o(image_size, bbox_size, n):
        #    return (bbox_size * n - image_size) * 1.0 / (n - 1.0)
        
        nx = _compute_n(w, size)
        ny = _compute_n(h, size)    
        #ox = _compute_o(w, size, nx)
        #oy = _compute_o(h, size, ny)    

        for i in range(nx * ny):
            x_index = i % nx
            y_index = int(np.floor(i * 1.0 / nx))
            #yield int(np.round(x_index * (size - ox))), int(np.round(y_index * (size - oy)))
            yield int(x_index * size), int(y_index * size)
            
    if not hasattr(K, 'image_data_format') and hasattr(K, 'image_dim_ordering'):
        data_format = "channels_last" if K.image_dim_ordering() == "tf" else "channels_first"                       
    else:
        data_format = K.image_data_format()
        
    # convert input image to channels_first format
    if channel_axis == 2:
        _image = image.transpose([2, 0, 1])
    else:
        _image = image
                                                    
    # define batch with 'channels_format'
    batch = np.zeros((batch_size,) + _image.shape, dtype=np.float32)
    _to_data_format = lambda x : x
    if data_format == 'channels_last':
        batch = batch.transpose([0, 2, 3, 1])
        _to_data_format = lambda image : image.transpose([1, 2, 0])

    # compute predictions on the original image:        
    batch[0, :, :, :] = _to_data_format(_image[:, :, :])
    y_pred_0 = model.predict_on_batch(batch[:1,:,:,:]) 
    
    heatmap = np.zeros((top_k_classes,) + _image.shape[1:])
    top_k_indices = np.argsort(y_pred_0[0, :])[:top_k_classes]
    for i in range(top_k_classes):
        heatmap[i, :, :] = y_pred_0[0, top_k_indices[i]]
        
    # compute predictions on images with black box 
    image_size = min(_image.shape[1], _image.shape[2])
    bbox_size = int(image_size * 2.0 ** ( -(resolution*4.0 + 1.0)))
               
    counter = 0
    batch_bbox_xy = []
    for x, y in _blackbox_coords_iterator(_image.shape[2], _image.shape[1], bbox_size):        
        image_copy = _image.copy()
        image_copy[:, y:y+bbox_size, x:x+bbox_size] = 0
        batch[counter, :, :, :] = _to_data_format(image_copy[:, :, :])
        batch_bbox_xy.append((x, y))
        counter += 1
        if counter == batch_size:
            y_pred = model.predict_on_batch(batch)
            for j, (_x, _y) in enumerate(batch_bbox_xy):
                for i in range(top_k_classes):
                    heatmap[i, _y:_y+bbox_size, _x:_x+bbox_size] -= y_pred[j, top_k_indices[i]]
            batch_bbox_xy = []
            counter = 0
               
    if counter > 0:
        y_pred = model.predict_on_batch(batch[:counter, :, :, :])
        for j, (_x, _y) in enumerate(batch_bbox_xy):
            for i in range(top_k_classes):
                heatmap[i, _y:_y+bbox_size, _x:_x+bbox_size] -= y_pred[j, top_k_indices[i]]
       
    heatmap[heatmap < 0.0] = 0.0
    return heatmap, top_k_indices






