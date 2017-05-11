#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLASSES = ['Type_1', 'Type_2', 'Type_3']
NUM_CLASSES = len(CLASSES)
NUM_SAMPLES = 512
PROBING_CLASS = 'Type_1'

if __name__ == "__main__":
    hand = pd.read_csv('hand.csv')[CLASSES].values

    true = np.zeros([NUM_SAMPLES, NUM_CLASSES])
    true[:, 0] = 1.0
    true[hand[:, CLASSES.index(PROBING_CLASS)] == 0.0, 0] = 0.0
    true[hand[:, CLASSES.index(PROBING_CLASS)] == 0.0, 1] = 1.0

    score = log_loss(true, hand)

    print('hand.csv submission must score %0.5f' % score)