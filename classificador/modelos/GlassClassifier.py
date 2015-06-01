import numpy as np
import os

from classificador.modelos.BaseClassifier import BaseClassifier

class GlassClassifier(BaseClassifier):
    datasetURI = os.path.join(os.getcwd(), "datasets/glass.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",")
        # print(dataset.shape)
        features = dataset[:, 1:10]
        labels = dataset[:, 10]
        return features, labels