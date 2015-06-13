import os

import numpy as np

from BaseClassifier import BaseClassifier

class WineClassifier(BaseClassifier):

    datasetURI = os.path.join(os.getcwd(), "datasets/wine.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",")
        # print(dataset.shape)
        features = dataset[:, 1:14]
        labels = dataset[:, 0]
        return features, labels
