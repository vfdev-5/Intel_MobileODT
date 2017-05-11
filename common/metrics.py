
import numpy as np


def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    https://github.com/ottogroup/kaggle/blob/master/benchmark.py

    y_true is ndarray of shape (n_samples, n_classes), one-hot encoded
    y_prob is ndarray of shape (n_samples, n_classes), probabilities
    """
    # normalize
    y_prob = y_prob / (y_prob.sum(axis=1).reshape(-1, 1) + epsilon) 
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, np.argmax(j)] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def jaccard_index(y_true, y_pred, epsilon=1e-15):
    """ Jaccard index
    """
    intersection = np.sum(y_true * y_pred, axis=(0, -1, -2))
    sum_ = np.sum(y_true + y_pred, axis=(0, -1, -2))
    jac = (intersection + epsilon*0.01) / (sum_ - intersection + epsilon)
    return np.mean(jac)
