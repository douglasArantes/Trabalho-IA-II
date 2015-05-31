import numpy as np
import os
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation as crossV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class DiabetesClassifier(object):
    __diabetesURI = os.path.join(os.getcwd(), "datasets/pima-indians-diabetes.txt")

    def __extractFeaturesAndLabels(self):
        dataset = np.loadtxt(self.__diabetesURI, delimiter=",")
        features = dataset[:, 0:8]
        labels = dataset[:, 8]
        return features, labels

    def __accuracyAndStd(self, model):
        features, labels = self.__extractFeaturesAndLabels()
        skf = StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=None)

        listAccuracysAndStd = []

        for trainIndex, testIndex in skf:
            # print("TRAIN:", train_index, "TEST:", test_index)
            featuresTrain, featuresTest = features[trainIndex], features[testIndex]
            labelsTrain, labelsTest = labels[trainIndex], labels[testIndex]
            model.fit(featuresTrain, labelsTrain)
            scores = crossV.cross_val_score(model, features, labels, scoring='accuracy')
            listAccuracysAndStd.append((scores.mean(), scores.std()))

        somaAcc = 0.0
        somaStd = 0.0
        for acc, std in listAccuracysAndStd:
            somaAcc += acc
            somaStd += std

        print("Accuracy: ", somaAcc / 10)
        print("Standard Deviation: ", somaStd / 10)

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