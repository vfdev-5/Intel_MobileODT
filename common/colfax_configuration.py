
import os
import sys


# On Colfax :
def setup_keras_122(check_mkl=False):
    if os.path.exists("/home/u2459/keras-1.2.2"):
        keras_lib_path = "/home/u2459/keras-1.2.2/build/lib"
        if not keras_lib_path in sys.path:
            sys.path.insert(0, "/home/u2459/keras-1.2.2/build/lib")
        from keras import __version__
        print("Keras version: ", __version__)
        import theano
        if check_mkl:
            print("mkl_available: ", theano.sandbox.mkl.mkl_available())

            
# On Colfax :
def setup_keras_202():
    path = "/home/u2459/.local/lib/python3.5/site-packages/"
    if os.path.exists(os.path.join(path, "theano")):
        print("Found a local Theano")
        sys.path.insert(0, path)     
    if os.path.exists(os.path.join(path, "keras")): 
        print("Found a local Keras")       
        if not path in sys.path:
            sys.path.insert(0, path)
        from keras import __version__
        print("Keras version: ", __version__)
        import theano
        print(theano.__version__)
        print (theano.numpy.show_config())
