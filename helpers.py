import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import csv
import cPickle as pickle
import os.path
import scipy.ndimage as nd
import pandas as pd
import random

TRAINING_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "train.csv")
TRAINING_SET_PICKLE_PATH = os.path.join(os.path.dirname(__file__), "pickles", "train.p")

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "test.csv")
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "data", "knn_benchmark.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "result.csv")


USE_PICKLE = False
IMAGE_WIDTH = 28

N_TRAINING_DIGITS = 42000

def chunks_of_training_data(num_chunks=2):
    chunk_id = 0
    step_size = N_TRAINING_DIGITS/num_chunks
    while chunk_id < num_chunks:
        offset = chunk_id*step_size
        yield load_training_digits(step_size, offset)
        chunk_id += 1

def load_training_digits(limit=np.inf, offset=0):
    offset = offset + 1
    X = []
    Y = []
    with open(TRAINING_SET_PATH, 'r') as f:
        i = 0
        for line in f.readlines():
            if i < offset:
                i+=1
                continue
            if i > offset + limit:
                return (np.array(X), np.array(Y))
            split_line = line.split(',', 1)
            classification = int(split_line[0][0])
            pixels_string = split_line[1]
            pixels = [int(pixel)/255.0 for pixel in pixels_string.split(",")]
            pixel_grid = [[pixels[y*28 + x] for x in range(0,28)] for y in range(0,28)]
            X.append(pixel_grid)
            Y.append(classification)
            i+=1
    image = np.array(X)
    target = np.array(Y)

    return (image, target)

def load_training_data():
    data = pd.DataFrame.as_matrix(pd.read_csv(TRAINING_SET_PATH))
    Y = data[:, 0]
    data = data[:, 1:] # trim first classification field
    X = data/255.0
    return X, Y

def images_to_data(images):
    return np.reshape(images,(len(images),-1))

def average(x):
    return sum(x)/len(x)

def compress_images(images):
    new_images = []
    print images[0]
    for image in images:
        print image
        new_image = [[average([image[y*4, x*4], image[y*4, x*4+1], image[y*4+1, x*4], image[y*4+1, x*4+1]]) for x in range(0,28/4)] for y in range(0,28/4)]
        new_images.append(new_image)
    return np.array(new_images)

def nudge_dataset(X, Y):
    nudge_size = 2
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

def rotate_dataset(X, Y):
    n_images = X.shape[0]
    for index in range(n_images):
        if index % 100 == 0:
            print index
        angle = random.uniform(np.pi*(-1/4.0), np.pi*(1/4.0))
        XX = nd.rotate(X[index, :], angle, reshape=False).ravel()
        YY = Y[index]
        print XX.shape
        X = np.hstack((X,XX))
        Y = np.hstack((Y,YY))
    return X, Y

def get_test_data_set():
    test_data = pd.read_csv(TEST_SET_PATH).values
    return test_data

def get_benchmark():
    return pd.read_csv(BENCHMARK_PATH)

def write_to_csv(subm):
    subm.to_csv(RESULTS_PATH, index_label='ImageId', col=['Label'], index=False)
