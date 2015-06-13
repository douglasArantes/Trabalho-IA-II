import os

import numpy as np

from BaseClassifier import BaseClassifier
#import BaseClassifier


class DiabetesClassifier(BaseClassifier):

    datasetURI = os.path.join(os.getcwd(), "datasets/pima-indians-diabetes.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",")
        features = dataset[:, 0:8]
        labels = dataset[:, 8]
        return features, labels