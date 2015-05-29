import numpy as np
import os

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class BalanceScale(object):

    irisUrl = os.path.join(os.getcwd(), "datasets/balance-scale.txt")

    def balanceScaleLabelToFloat(self, bstr):
        str = bstr.decode("utf-8")
        if str == "B":
            return 0.0
        if str == "R":
            return 1.0
        elif str == "L":
            return 2.0

    #@staticmethod
    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.irisUrl, delimiter=",", converters={0: lambda s: self.balanceScaleLabelToFloat(s)})
        #print(dataset.shape)
        features = dataset[:, 1:4]
        labels = dataset[:, 0]
        return features, labels

    def printInfo(self, expected, predicted):
        print("Relatório de Classificação")
        print(metrics.classification_report(expected, predicted))
        print("Matriz de Confusão")
        print(metrics.confusion_matrix(expected, predicted))

    def knnOnBalanceScaleDataset(self):
        print("\nAlgoritmos KNN - Iris Dataset")
        features, labels = self.extractFeaturesAndLabels()

        model = KNeighborsClassifier()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)

    def naiveBayesOnBalanceScaleDataset(self):
        print("\nAlgoritmos Naive Bayes - Iris Dataset")
        features, labels = self.extractFeaturesAndLabels()

        model = GaussianNB()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)

    def decisionTreeOnBalanceScaleDataset(self):
        print("\nAlgoritmos Decision Tree - Balance Scale Dataset")
        features, labels = self.extractFeaturesAndLabels()

        model = DecisionTreeClassifier()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)