# These are all the libraries that will be probably used across different models
from keras.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def importCifar10(type = "split"):
    """
    Imports dataset and returns dataset based on type.
    If type = 'split', then returns two pairs of tuples:
        1. Training X and Y
        2. Testing X and Y
    If type = 'combined', then returns one pair of tuples
    that doesn't split the dataset in half already.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if type == "split": 
        return (x_train, y_train), (x_test, y_test)
    elif type == "combined":
        x = np.vstack((x_train, x_test))
        y = np.vstack((y_train,y_test))
        return (x,y)
    else: 
        return None

