import numpy as np
import os
from classificador.modelos.BaseClassifier import BaseClassifier


class IonosphereClassifier(BaseClassifier):
    datasetURI = os.path.join(os.getcwd(), "datasets/ionosphere.txt")

    def labelToFloat(self, bstr):
        str = bstr.decode("utf-8")
        if str == "b":
            return 0.0
        elif str == "g":
            return 1.0

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",", converters={34: lambda s: self.labelToFloat(s)})
        # print(dataset.shape)
        features = dataset[:, 0:34]
        labels = dataset[:, 34]
        return features, labels