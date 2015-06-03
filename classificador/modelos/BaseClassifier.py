import os

from sklearn import cross_validation as crossV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class BaseClassifier(object):

    datasetURI = None

    def labelToFloat(self, bstr):
        pass

    def extractFeaturesAndLabels(self):
        raise NotImplementedError

    def accuracyAndStd(self, model, fileWriter):
        features, labels = self.extractFeaturesAndLabels()
        scores = crossV.cross_val_score(model, features, labels, cv=10, scoring='accuracy')

        fileWriter.write("Acurácia dos 10 folds:\n")

        print("Acurácia dos 10 folds:")
        for score in scores:
            fileWriter.write(str(score) + "\n")
            print(score)

        fileWriter.write("Média das Acurácias %s" % scores.mean() + "\n")
        fileWriter.write("Desvio Padrão das Acurácias %s" % scores.std() + "\n")
        print("Média das Acurácias %s" % scores.mean())
        print("Desvio Padrão das Acurácias %s" % scores.std())

    def printAccruraciesAndStds(self, fileWriter):
        decisionTreeModel = DecisionTreeClassifier()

        knnModel1 = KNeighborsClassifier(n_neighbors=1)
        knnModel3 = KNeighborsClassifier(n_neighbors=3)
        knnModel4 = KNeighborsClassifier(n_neighbors=4)
        knnModel5 = KNeighborsClassifier(n_neighbors=5)

        naiveBayesModel = GaussianNB()

        fileWriter.write("\nDecision Tree Model\n\n")
        print("Decision Tree Model")
        self.accuracyAndStd(decisionTreeModel, fileWriter)

        fileWriter.write("\nKNN Model K = 1\n\n")
        print("\nKNN Model K = 1")
        self.accuracyAndStd(knnModel1, fileWriter)

        fileWriter.write("\nKNN Model K = 3\n\n")
        print("\nKNN Model K = 3")
        self.accuracyAndStd(knnModel3, fileWriter)

        fileWriter.write("\nKNN Model K = 4\n\n")
        print("\nKNN Model K = 4")
        self.accuracyAndStd(knnModel4, fileWriter)

        fileWriter.write("\nKNN Model K = 5\n\n")
        print("\nKNN Model K = 5")
        self.accuracyAndStd(knnModel5, fileWriter)

        fileWriter.write("\nNaive Bayes Model\n\n")
        print("\nNaive Bayes Model")
        self.accuracyAndStd(naiveBayesModel, fileWriter)