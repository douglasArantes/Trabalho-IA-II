from classificador.teste_online_datasets.IrisClassifier import IrisClassifier
from classificador.teste_online_datasets.DiabetesClassifier import DiabetesClassifier
from classificador.teste_online_datasets.BalanceScaleClassifier import BalanceScaleClassifier
from classificador.teste_online_datasets.SonarClassifier import SonarClassifier
from classificador.teste_online_datasets.GlassClassifier import GlassClassifier
from classificador.teste_online_datasets.HabermanClassifier import HabermanClassifier
from classificador.teste_online_datasets.IonosphereClassifier import IonosphereClassifier
from classificador.teste_online_datasets.SpamClassifier import SpamClassifier
from classificador.teste_online_datasets.ImageSegmentationClassifier import ImageSegmentationClassifier
from classificador.teste_online_datasets.WineClassifier import WineClassifier

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

        print("-------- Dataset Sonar All --------\n")
        SonarClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Glass Sonar All --------\n")
        GlassClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Haberman Sonar All --------\n")
        HabermanClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Ionsphere --------\n")
        IonosphereClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Spams --------\n")
        SpamClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Image Segmentation --------\n")
        ImageSegmentationClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Wine --------\n")
        WineClassifier().printAccruraciesAndStds()
        print()