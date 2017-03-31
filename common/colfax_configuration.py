
import os
import sys


# On Colfax :
def setup_keras_122():
    if os.path.exists("/home/u2459/keras-1.2.2"):
        keras_lib_path = "/home/u2459/keras-1.2.2/build/lib"
        if not keras_lib_path in sys.path:
            sys.path.insert(0, "/home/u2459/keras-1.2.2/build/lib")
        from keras import __version__
        print "Keras version: ", __version__
        import theano
        print "mkl_available: ", theano.sandbox.mkl.mkl_available()
