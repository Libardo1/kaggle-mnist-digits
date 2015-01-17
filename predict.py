import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import svm, metrics
from helpers import load_training_digits, images_to_data, nudge_dataset, rotate_dataset, compress_images, get_test_data_set, get_benchmark, write_to_csv, chunks_of_training_data
import numpy as np
import sklearn.decomposition as deco
import pandas as pd

IMAGE_WIDTH = 28

def run():
    classifier = svm.SVC(C=2.8, gamma=0.0073, kernel='rbf')
    i = 0
    for images, target in chunks_of_training_data():
        print "training on chunk", i
        i += 1
        X_train = images_to_data(images)
        Y_train = target

        X_train, Y_train = nudge_dataset(X_train, Y_train)

        classifier.fit(X_train, Y_train)


    test_data = get_test_data_set()
    test_data = np.asarray(test_data / 255.0, 'float32')
    subm = get_benchmark()
    subm.Label = classifier.predict(test_data)
    write_to_csv(subm)

def __main__(args):
    run()

if __name__ == "__main__":
    __main__(sys.argv)
