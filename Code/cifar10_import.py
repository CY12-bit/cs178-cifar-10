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


def flattenImageSet(images, gray = False):
    col_size = 3072 if gray == False else 1024
    images = images.flatten().reshape(len(images),col_size)
    return images

def scaleImages(images, scalar = None):
    if scalar == None:
        scalar = StandardScaler()
        x_scaled = scalar.fit_transform(images)
        return (x_scaled,scalar)
    else:
        images = scalar.transform(images)
        return images

"""
def prepareX(x, x_test = None, x_val = None, gray_scale = False, scaled = True):
    col_size = 3072 if gray_scale == False else 1024

    x_scaled = x.flatten().reshape(len(x),col_size)
    if scaled == True:
        scalar = StandardScaler()
        x_scaled = scalar.fit_transform(x_scaled)
        return (x_scaled, scalar.transform(x_test.flatten().reshape(len(x_test),col_size)),scalar) if type(x_test) != type(None) else (x_scaled,scalar)
    else:
        return (x_scaled, x_test.flatten().reshape(len(x_test),col_size),None) if type(x_test) != type(None) else (x_scaled,None)
"""

def grayScaleData(*args):
    """
    Function turns a colored image array into a gray-scaled image array
    """
    return (
        np.array([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in x_images]) \
        for x_images in args
    )

# Function displays images and corresponding labels from CIFAR-10 dataset
# Can specify the number of images and if it's a certain class
def displayImages(n:int,x,y,c=None):
    plt.figure(figsize=(14,14))
    if c != None:
        indices = np.where(y == c)
        x = x[indices]
        y=y[indices]
    for i in range(min(n,len(y))):
        plt.subplot(8, 8, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        c = plt.imshow(x[i])

        # Set the label as the title
        plt.title(class_names[y[i]], fontsize=10)
    
    plt.tight_layout()
