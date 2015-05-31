import numpy as np
import os
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation as crossV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class IrisClassifier(object):
    __irisURI = os.path.join(os.getcwd(), "datasets/iris.txt")

    def __irisLabelToFloat(self, bstr):
        str = bstr.decode("utf-8")
        if str == "Iris-setosa":
            return 0.0
        if str == "Iris-versicolor":
            return 1.0
        else:
            return 2.0

    def __extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.__irisURI, delimiter=",", converters={4: lambda s: self.__irisLabelToFloat(s)})
        # print(dataset.shape)
        features = dataset[:, 0:4]
        labels = dataset[:, 4]
        return features, labels

    def __accuracyAndStd(self, model):
        features, labels = self.__extractFeaturesAndLabels()
        scores = crossV.cross_val_score(model, features, labels, cv=10, scoring='accuracy')

        print("Accuracy of 10 folds:")
        for score in scores:
            print(score)
        print("Standard Deviation of Accuracy %s" % scores.std())

    def printAccruraciesAndStds(self):
        decisionTreeModel = DecisionTreeClassifier()
        knnModel = KNeighborsClassifier()
        naiveBayesModel = GaussianNB()

        print("Decision Tree Model")
        self.__accuracyAndStd(decisionTreeModel)
        print("\nKNN Model")
        self.__accuracyAndStd(knnModel)
        print("\nNaive Bayes Model")
        self.__accuracyAndStd(naiveBayesModel)

"""
 def __accuracyAndStd(self, classifier):
        features, labels = self.__extractFeaturesAndLabels()
        skf = StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=None)

        listAccuracysAndStd = []

        for trainIndex, testIndex in skf:
            #print("TRAIN:", trainIndex, "TEST:", testIndex)
            featuresTrain, featuresTest = features[trainIndex], features[testIndex]
            labelsTrain, labelsTest = labels[trainIndex], labels[testIndex]
            model = classifier.fit(featuresTrain, labelsTrain)
            scores = crossV.cross_val_score(model, featuresTest, labelsTest, scoring='accuracy')
            listAccuracysAndStd.append((scores.mean(), scores.std()))

        somaAcc = 0.0
        somaStd = 0.0
        for acc, std in listAccuracysAndStd:
            somaAcc += acc
            somaStd += std

        print("Accuracy: ", somaAcc / 10)
        print("Standard Deviation: ", somaStd / 10)
"""