from classificador.modelos.BaseClassifier import BaseClassifier
import numpy as np
import os

class IrisClassifier(BaseClassifier):

    datasetURI = os.path.join(os.getcwd(), "datasets/iris.txt")

    def labelToFloat(self, bstr):
        str = bstr.decode("utf-8")
        if str == "Iris-setosa":
            return 0.0
        if str == "Iris-versicolor":
            return 1.0
        else:
            return 2.0

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",", converters={4: lambda s: self.labelToFloat(s)})
        # print(dataset.shape)
        features = dataset[:, 0:4]
        labels = dataset[:, 4]
        return features, labels


