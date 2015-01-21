import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import svm, metrics
from helpers import load_training_digits, images_to_data, nudge_dataset, rotate_dataset, compress_images, get_test_data_set, get_benchmark, write_to_csv, chunks_of_training_data, load_training_data
import numpy as np
import sklearn.decomposition as deco
import pandas as pd
from sklearn import linear_model
from nolearn.dbn import DBN

IMAGE_WIDTH = 28

def run():
    X, Y = load_training_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.1,
                                                            random_state=0)

    # X_train = X
    # Y_train = Y

    X_train, Y_train = nudge_dataset(X_train, Y_train)
    print "nudge done"
    # X_train, Y_train = rotate_dataset(X_train, Y_train)
    print "rotation done"

    n_features = X_train.shape[1]
    n_classes = 10
    classifier = DBN([n_features, 300, n_classes], learn_rates=0.3, learn_rate_decays=0.9 ,epochs=25, verbose=1)

    classifier.fit(X_train, Y_train)

    acc_nn = classifier.score(X_test,Y_test)
    print "acc_n", acc_nn

    test_data = get_test_data_set()
    test_data = np.asarray(test_data / 255.0, 'float32')
    subm = get_benchmark()
    subm.Label = classifier.predict(test_data)
    write_to_csv(subm)

def __main__(args):
    run()

if __name__ == "__main__":
    __main__(sys.argv)
