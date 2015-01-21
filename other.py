import matplotlib.pyplot as plt
import sys
from sklearn.cross_validation import train_test_split
from sklearn import datasets, svm, metrics
from helpers import load_training_digits, images_to_data, nudge_dataset, rotate_dataset, compress_images
import numpy as np
import sklearn.decomposition as deco
import pandas as pd

IMAGE_WIDTH = 28

np.set_printoptions(threshold=np.nan)

def run():
    images, target = load_training_digits(100)
    X = images_to_data(images)
    Y = target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.5,
                                                            random_state=0)

    X_train, Y_train = nudge_dataset(X_train, Y_train)
    X_train, Y_train = rotate_dataset(X_train, Y_train)

    images_and_labels = list(zip(images, Y))
    faw = images_and_labels[:2] + images_and_labels[100:102]
    for index, (image, label) in enumerate(faw):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)

    classifier = svm.SVC(gamma=0.001, kernel='rbf')
    classifier.fit(X_train, Y_train)

    expected = Y_test
    predicted = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    images_and_predictions = list(zip(X_test, predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image.reshape((IMAGE_WIDTH, IMAGE_WIDTH)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)

    plt.show()

def __main__(args):
    run()

if __name__ == "__main__":
    __main__(sys.argv)