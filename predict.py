import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import svm, metrics
from helpers import images_to_data, nudge_dataset, rotate_dataset
from helpers import compress_images, get_test_data_set, write_predictions_to_csv
from helpers import load_training_data
import numpy as np
import sklearn.decomposition as deco
import pandas as pd
from sklearn import linear_model
from nolearn.dbn import DBN

def run():
    X_train, Y_train = load_training_data()

    X_train, Y_train = rotate_dataset(X_train, Y_train, 8)
    X_train, Y_train = nudge_dataset(X_train, Y_train)

    n_features = X_train.shape[1]
    n_classes = 10
    classifier = DBN([n_features, 8000, n_classes], 
        learn_rates=0.4, learn_rate_decays=0.9 ,epochs=75, verbose=1)

    classifier.fit(X_train, Y_train)

    test_data = get_test_data_set()
    predictions = classifier.predict(test_data)
    write_predictions_to_csv(predictions)

def __main__(args):
    run()

if __name__ == "__main__":
    __main__(sys.argv)
