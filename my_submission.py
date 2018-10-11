
import numpy as np
from tensorflow.contrib import keras

def main_tester():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("x_train")
    print(x_train)
    print("y_train")
    print(y_train)
    print("x_test")
    print(x_test)
    print("y_test")
    print(y_test)
    return





