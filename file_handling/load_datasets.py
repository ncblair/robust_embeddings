import tensorflow as tf
import numpy as np

def load_mnist():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    X_train = x_train.reshape((x_train.shape[0], -1))
    Y_train = np.eye(10)[y_train]

    X_test = x_test.reshape((x_test.shape[0], -1))
    Y_test = np.eye(10)[y_test]

    return (X_train, Y_train), (X_test, Y_test)
    