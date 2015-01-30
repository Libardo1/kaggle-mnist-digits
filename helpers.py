import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import csv
import cPickle as pickle
import os.path
import scipy.ndimage as nd
import pandas as pd
import random
import scipy
import time

TRAINING_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "train.csv")
TRAINING_SET_PICKLE_PATH = os.path.join(os.path.dirname(__file__), "pickles", "train.p")

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "test.csv")
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "data", "knn_benchmark.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "result.csv")

USE_PICKLE = False
IMAGE_WIDTH = 28

def normalize_data(X):
    X = X/255.0
    return X

def load_training_data():
    data = pd.DataFrame.as_matrix(pd.read_csv(TRAINING_SET_PATH))
    Y = data[:, 0]
    data = data[:, 1:] # trim first classification field
    X = normalize_data(data)
    return X, Y

def images_to_data(images):
    return np.reshape(images,(len(images),-1))

def average(x):
    return sum(x)/len(x)

def compress_images(images):
    new_images = []
    print images[0]
    for image in images:
        new_image = [[average([image[y*4, x*4], image[y*4, x*4+1], image[y*4+1, x*4], image[y*4+1, x*4+1]]) for x in range(0,28/4)] for y in range(0,28/4)]
        new_images.append(new_image)
    return np.array(new_images)

def nudge_dataset(X, Y):
    nudge_size = 1
    direction_matricies = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    scaled_direction_matricies = [[[comp*nudge_size for comp in vect] for vect in matrix] for matrix in direction_matricies]
    shift = lambda x, w: convolve(x.reshape((IMAGE_WIDTH, IMAGE_WIDTH)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in scaled_direction_matricies])

    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

def rotate_dataset(X, Y, n_rotations=2):
    for rot_i in range(n_rotations):
        rot_shape = (X.shape[0], X.shape[1])
        rot_X = np.zeros(rot_shape)
        for index in range(X.shape[0]):
            sign = random.choice([-1, 1])
            angle = np.random.randint(1, 12)*sign
            rot_X[index, :] = threshold(nd.rotate(np.reshape(X[index, :], ((IMAGE_WIDTH, IMAGE_WIDTH))), angle, reshape=False).ravel())
        XX = np.vstack((X,rot_X))
        YY = np.hstack((Y,Y))
    return XX, YY

def threshold(X):
    X[X < 0.1] = 0.0
    X[X >= 0.9] = 1.0
    return X

def sigmoid(X):
    return scipy.special.expit(X)

def get_test_data_set():
    data = pd.DataFrame.as_matrix(pd.read_csv(TEST_SET_PATH))
    X = normalize_data(data)
    return X

def get_benchmark():
    return pd.read_csv(BENCHMARK_PATH)

def get_time_hash():
    return str(int(time.time()))

def make_predictions_path():
    base_string = "predictions"
    file_name = base_string + "-" + get_time_hash() + ".csv"
    file_path =  os.path.join(os.path.dirname(__file__), "data", file_name)
    return file_path

def write_predictions_to_csv(predictions):
    csv_path = make_predictions_path()
    predictions_dict = {"ImageId": range(1, len(predictions)+1), "Label": predictions}
    predictions_table = pd.DataFrame(predictions_dict)
    predictions_table.to_csv(csv_path, index=False)
