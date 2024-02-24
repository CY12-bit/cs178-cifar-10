# These are all the libraries that will be probably used across different models
from keras.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
        return (x_train, np.array([y[0] for y in y_train])), (x_test, np.array([y[0] for y in y_test]))
    elif type == "combined":
        x = np.vstack((x_train, x_test))
        y = np.vstack((y_train,y_test))
        return (x,np.array([y[0] for y in y]).T)
    else: 
        return None

def prepareX(x, x_test = None, gray_scale = False, scaled = True):
    """
    Function flattens x to be a 2d array and conditionally standardizes it.
    """
    col_size = 3072 if gray_scale == False else 1024

    x_scaled = x.flatten().reshape(len(x),col_size)
    if scaled == True:
        scalar = StandardScaler()
        x_scaled = scalar.fit_transform(x_scaled)
        return (x_scaled, scalar.transform(x_test.flatten().reshape(len(x_test),col_size))) if type(x_test) != type(None) else x_scaled
    else:
        return (x_scaled, x_test.flatten().reshape(len(x_test),col_size)) if type(x_test) != type(None) else x_scaled

def grayScaleData(*args):
    """
    Function turns a colored image array into a gray-scaled image array
    """
    return (
        np.array([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in x_images]) \
        for x_images in args
    )

def printFirst64(x,y):
    plt.figure(figsize=(15,15))
    plt.rcParams['image.cmap'] = 'gray'
    # Loop over the first 64 images
    for i in range(64):
        # Create a subplot for each image
        
        plt.subplot(8, 8, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        c = plt.imshow(x[i])

        # Set the label as the title
        plt.title(class_names[y[i]], fontsize=12)