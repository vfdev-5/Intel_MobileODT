# How to use this code

**At first, make sure that you have all necessary dependencies**

- Python 2.7 and Python 3.5
- Keras versions 1.2.2 and 2.0.2
- Numpy, Pandas, opencv-python
- Scikit-image, Scikit-learn
- etc

We assume that input data is handled as test dataset: a folder with 'jpg' images

**Put data in the folder `<REPO>/input` and name test data folder as `test`**


## TL;DR

Run the following script, take a coffee and wait until the end:
```
sh start.sh
```
However, if your configuration is not set properly, script may fail. When it is finished successfully, you can
find results in csv file `<REPO/results/submission_final_classification.csv`.

Below you can find some information about what is done.


## 1. Cervix detection

At first we need to detect cervix part on test image. We use Python 2.7 and Keras 1.2.2 for this script with Theano backend.
Run the following command :

```
python 1_cervix_os_detection.py
```

The following script loads pretrained U-Net models and generates cervix/os masks and bounding boxes.
Results are stored in `<REPO>/input/generated`.


## 2. Cervix classification

Now we can use previously obtained detections to predict cervix types.
We use Python 3.5 and Keras 2.0.2 for this script with Theano backend. Run the following command :

```
python3 2a_cervix_os_classification.py
python3 2b_cervix_os_classification.py
```

This script loads pretrained models (Custom SqueezeNet, Custom VGG and others) and computes probabilities of cervix type.
Results are stored in a csv file in `<REPO>/results`.

## 3. Merge results

```
python 3_merge_classifications.py
```
Results are stored in a csv file in `<REPO/results/submission_final_classification.csv`.
