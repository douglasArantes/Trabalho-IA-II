import numpy as np
import os

from classificador.modelos.BaseClassifier import BaseClassifier


class SpamClassifier(BaseClassifier):
    datasetURI = os.path.join(os.getcwd(), "datasets/spambase.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",")
        # print(dataset.shape)
        features = dataset[:, 0:57]
        labels = dataset[:, 57]
        return features, labels
