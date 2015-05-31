from classificador.teste_online_datasets.IrisClassifier import IrisClassifier
from classificador.teste_online_datasets.DiabetesClassifier import DiabetesClassifier
from classificador.teste_online_datasets.BalanceScaleClassifier import BalanceScaleClassifier

class Teste:
    if __name__ == "__main__":

        print("-------- Dataset Iris --------\n")
        IrisClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Prima Indians Diabetes --------\n")
        DiabetesClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Balance Scale --------\n")
        BalanceScaleClassifier().printAccruraciesAndStds()
        print()


