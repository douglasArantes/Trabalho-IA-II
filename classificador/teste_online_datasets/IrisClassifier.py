import numpy as np
import os

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class IrisClassifier(object):
    irisUrl = os.path.join(os.getcwd(), "datasets/iris.txt")

    def irisLabelToFloat(self, bstr):
        str = bstr.decode("utf-8")
        if str == "Iris-setosa":
            return 0.0
        if str == "Iris-versicolor":
            return 1.0
        else:
            return 2.0

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.irisUrl, delimiter=",", converters={4: lambda s: self.irisLabelToFloat(s)})
        # print(dataset.shape)
        features = dataset[:, 0:3]
        labels = dataset[:, 4]
        return features, labels

    def printInfo(self, expected, predicted):
        print("Relatório de Classificação")
        print(metrics.classification_report(expected, predicted))
        print("Matriz de Confusão")
        print(metrics.confusion_matrix(expected, predicted))

    def knnOnIrisDataset(self):
        print("\nAlgoritmos KNN - Iris Dataset")

        features, labels = self.extractFeaturesAndLabels()

        model = KNeighborsClassifier()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)

    def naiveBayesOnIrisDataset(self):
        print("\nAlgoritmos Naive Bayes - Iris Dataset")
        features, labels = self.extractFeaturesAndLabels()

        model = GaussianNB()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)

    def decisionTreeOnIrisDataset(self):
        print("\nAlgoritmos Decision Tree - Iris Dataset")
        features, labels = self.extractFeaturesAndLabels()

        model = DecisionTreeClassifier()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)