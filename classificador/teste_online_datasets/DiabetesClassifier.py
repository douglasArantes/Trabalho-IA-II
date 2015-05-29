import numpy as np
import os

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class DiabetesClassifier(object):
    diabetesUrl = os.path.join(os.getcwd(), "datasets/pima-indians-diabetes.txt")
    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.diabetesUrl, delimiter=",")
        features = dataset[:, 0:7]
        labels = dataset[:, 8]
        return features, labels

    def knnOnDiabetesDataset(self):

        print("\nAlgoritmos KNN - Diabetes Dataset")
        features, labels = self.extractFeaturesAndLabels()

        model = KNeighborsClassifier()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)

    def naiveBayesOnDiabetesDataset(self):
        print("\nAlgoritmos Naive Bayes - Diabetes Dataset")
        features, labels = self.extractFeaturesAndLabels()

        model = GaussianNB()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)

    def printInfo(self, expected, predicted):
        print("Relatório de Classificação")
        print(metrics.classification_report(expected, predicted))
        print("Matriz de Confusão")
        print(metrics.confusion_matrix(expected, predicted))

    def decisionTreeOnDiabetesDataset(self):
        print("\nAlgoritmos Decision Tree - Diabetes Dataset")

        features, labels = self.extractFeaturesAndLabels()

        model = DecisionTreeClassifier()
        model.fit(features, labels)
        #print(model)

        expected = labels
        predicted = model.predict(features)
        self.printInfo(expected, predicted)