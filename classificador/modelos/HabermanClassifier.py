import numpy as np
import os
from classificador.modelos.BaseClassifier import BaseClassifier


class HabermanClassifier(BaseClassifier):
    datasetURI = os.path.join(os.getcwd(), "datasets/haberman.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",")
        # print(dataset.shape)
        features = dataset[:, 0:3]
        labels = dataset[:, 3]
        return features, labels

