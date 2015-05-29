from classificador.teste_online_datasets.IrisClassifier import IrisClassifier
from classificador.teste_online_datasets.DiabetesClassifier import DiabetesClassifier
from classificador.teste_online_datasets.BalanceScale import BalanceScale

class Teste:
    if __name__ == "__main__":

        print("DATASET IRIS")
        BalanceScale().decisionTreeOnBalanceScaleDataset()
        BalanceScale().knnOnBalanceScaleDataset()
        BalanceScale().naiveBayesOnBalanceScaleDataset()
        print()

        """
        #print("DATASET IRIS")
        #IrisClassifier().decisionTreeOnIrisDataset()
        #IrisClassifier().knnOnIrisDataset()
        #IrisClassifier().naiveBayesOnIrisDataset()
        #print()

        print("DATASET DIABETES")
        DiabetesClassifier().decisionTreeOnDiabetesDataset()
        DiabetesClassifier().knnOnDiabetesDataset()
        DiabetesClassifier().naiveBayesOnDiabetesDataset()
        print()
        """




