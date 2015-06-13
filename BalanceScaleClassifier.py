import os

import numpy as np

from BaseClassifier import BaseClassifier
#import BaseClassifier


class BalanceScaleClassifier(BaseClassifier):

    datasetURI = os.path.join(os.getcwd(), "datasets/balance-scale.txt")

    def labelToFloat(self, bstr):
        str = bstr.decode("utf-8")
        if str == "B":
            return 0.0
        if str == "R":
            return 1.0
        elif str == "L":
            return 2.0

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=",", converters={0: lambda s: self.labelToFloat(s)})
        #print(dataset.shape)
        features = dataset[:, 1:5]
        labels = dataset[:, 0]
        return features, labels