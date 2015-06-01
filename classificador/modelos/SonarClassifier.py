import numpy as np
import os

from classificador.modelos.BaseClassifier import BaseClassifier

class SonarClassifier(BaseClassifier):
    datasetURI = os.path.join(os.getcwd(), "datasets/sonar_all.txt")

    def labelToFloat(self, bstr):
        str = bstr.decode("utf-8")
        if str == "R":
            return 0.0
        elif str == "M":
            return 1.0

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",", converters={60: lambda s: self.labelToFloat(s)})
        # print(dataset.shape)
        features = dataset[:, 0:60]
        labels = dataset[:, 60]
        return features, labels