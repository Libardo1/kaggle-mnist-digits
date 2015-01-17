# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from helpers import load_training_digits, images_to_data, nudge_dataset, rotate_dataset, compress_images, get_test_data_set, get_benchmark, write_to_csv
import numpy as np
import sklearn.decomposition as deco
import pandas as pd

IMAGE_WIDTH = 28


np.set_printoptions(threshold=np.nan)

def run():
    digits = datasets.load_digits()
    images = digits.images
    target = digits.target

    images, target = load_training_digits(1000)


    X = images_to_data(images)

    Y = target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.5,
                                                            random_state=0)

    X_train, Y_train = X, Y
    X_train, Y_train = nudge_dataset(X_train, Y_train)
    #X_train, Y_train = rotate_dataset(X_train, X_train)

    classifier = svm.SVC(C=2.8, gamma=0.0073, kernel='rbf')

    classifier.fit(X_train, Y_train)

    expected = Y_test
    predicted = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    test_data = get_test_data_set()
    test_data = np.asarray(test_data / 255.0, 'float32')
    subm = get_benchmark()
    subm.Label = classifier.predict(test_data)
    write_to_csv(subm)


def __main__(args):
    run()

if __name__ == "__main__":
    __main__(sys.argv)