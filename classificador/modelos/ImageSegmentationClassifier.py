import numpy as np
import os

from classificador.modelos.BaseClassifier import BaseClassifier


class ImageSegmentationClassifier(BaseClassifier):
    datasetURI = os.path.join(os.getcwd(), "datasets/img-segment.txt")

    def extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.datasetURI, delimiter=" ")
        # print(dataset.shape)
        features = dataset[:, 0:19]
        labels = dataset[:, 19]
        return features, labels
