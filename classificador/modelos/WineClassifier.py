import numpy as np
import os

from sklearn import cross_validation as crossV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from classificador.modelos.BaseClassifier import BaseClassifier


class WineClassifier(BaseClassifier):

    datasetURI = os.path.join(os.getcwd(), "datasets/wine.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",")
        # print(dataset.shape)
        features = dataset[:, 1:14]
        labels = dataset[:, 0]
        return features, labels
