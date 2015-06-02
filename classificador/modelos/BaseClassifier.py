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

    def accuracyAndStd(self, model):
        features, labels = self.extractFeaturesAndLabels()
        scores = crossV.cross_val_score(model, features, labels, cv=10, scoring='accuracy')

        print("Acurácia dos 10 folds:")
        for score in scores:
            print(score)
        print("Média das Acurácias %s" % scores.mean())
        print("Desvio Padrão das Acurácias %s" % scores.std())

    def printAccruraciesAndStds(self):
        decisionTreeModel = DecisionTreeClassifier()

        knnModel1 = KNeighborsClassifier(n_neighbors=1)
        knnModel3 = KNeighborsClassifier(n_neighbors=3)
        knnModel5 = KNeighborsClassifier(n_neighbors=5)

        naiveBayesModel = GaussianNB()

        print("Decision Tree Model")
        self.accuracyAndStd(decisionTreeModel)
        print("\nKNN Model K = 1")
        self.accuracyAndStd(knnModel1)
        print("\nKNN Model K = 3")
        self.accuracyAndStd(knnModel3)
        print("\nKNN Model K = 5")
        self.accuracyAndStd(knnModel5)
        print("\nNaive Bayes Model")
        self.accuracyAndStd(naiveBayesModel)