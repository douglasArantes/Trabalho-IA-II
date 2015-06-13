import os

import numpy as np

from BaseClassifier import BaseClassifier
#import BaseClassifier


class BanknoteAuthenticationClassifier(BaseClassifier):

    datasetURI = os.path.join(os.getcwd(), "datasets/banknote_authentication.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",")

        # print(dataset.shape)
        features = dataset[:, 0:4]
        labels = dataset[:, 4]
        return features, labels