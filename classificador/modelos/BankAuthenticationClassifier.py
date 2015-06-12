from classificador.modelos.BaseClassifier import BaseClassifier
import numpy as np
import os

class BanknoteAuthenticationClassifier(BaseClassifier):

    datasetURI = os.path.join(os.getcwd(), "datasets/banknote_authentication.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",")

        # print(dataset.shape)
        features = dataset[:, 0:4]
        labels = dataset[:, 4]
        return features, labels