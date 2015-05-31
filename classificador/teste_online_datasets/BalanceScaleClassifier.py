import numpy as np
import os
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation as crossV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class BalanceScaleClassifier(object):

    __irisURI = os.path.join(os.getcwd(), "datasets/balance-scale.txt")

    def __balanceScaleLabelToFloat(self, bstr):
        str = bstr.decode("utf-8")
        if str == "B":
            return 0.0
        if str == "R":
            return 1.0
        elif str == "L":
            return 2.0

    def __extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.__irisURI, delimiter=",", converters={0: lambda s: self.__balanceScaleLabelToFloat(s)})
        #print(dataset.shape)
        features = dataset[:, 1:5]
        labels = dataset[:, 0]
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