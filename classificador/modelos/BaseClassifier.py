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

        print("Accuracy of 10 folds:")
        for score in scores:
            print(score)
        print("Standard Deviation of Accuracy %s" % scores.std())

    def printAccruraciesAndStds(self):
        decisionTreeModel = DecisionTreeClassifier()
        knnModel = KNeighborsClassifier()
        naiveBayesModel = GaussianNB()

        print("Decision Tree Model")
        self.accuracyAndStd(decisionTreeModel)
        print("\nKNN Model")
        self.accuracyAndStd(knnModel)
        print("\nNaive Bayes Model")
        self.accuracyAndStd(naiveBayesModel)