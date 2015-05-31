import numpy as np
import os

from sklearn import cross_validation as crossV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class ImageSegmentationClassifier(object):
    __imgsegURI = os.path.join(os.getcwd(), "datasets/img-segment.txt")

    def __extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.__imgsegURI, delimiter=" ")
        # print(dataset.shape)
        features = dataset[:, 0:19]
        labels = dataset[:, 19]
        return features, labels

    def __accuracyAndStd(self, model):
        features, labels = self.__extractFeaturesAndLabels()
        scores = crossV.cross_val_score(model, features, labels, cv=10, scoring='accuracy')

        print("Accuracy of 10 folds:")
        for score in scores:
            print(score)
        print("Standard Deviation of Accuracy %s" % scores.std())

    def printAccruraciesAndStds(self):
        decisionTreeModel = DecisionTreeClassifier()
        knnModel = KNeighborsClassifier()
        naiveBayesModel = GaussianNB()

        print("Decision Tree Model")
        self.__accuracyAndStd(decisionTreeModel)
        print("\nKNN Model")
        self.__accuracyAndStd(knnModel)
        print("\nNaive Bayes Model")
        self.__accuracyAndStd(naiveBayesModel)