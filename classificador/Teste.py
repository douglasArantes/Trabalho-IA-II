from classificador.modelos.IrisClassifier import IrisClassifier
from classificador.modelos.DiabetesClassifier import DiabetesClassifier
from classificador.modelos.BalanceScaleClassifier import BalanceScaleClassifier
from classificador.modelos.SonarClassifier import SonarClassifier
from classificador.modelos.GlassClassifier import GlassClassifier
from classificador.modelos.HabermanClassifier import HabermanClassifier
from classificador.modelos.IonosphereClassifier import IonosphereClassifier
from classificador.modelos.SpamClassifier import SpamClassifier
from classificador.modelos.ImageSegmentationClassifier import ImageSegmentationClassifier
from classificador.modelos.WineClassifier import WineClassifier

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

        print("-------- Dataset Glass  --------\n")
        GlassClassifier().printAccruraciesAndStds()
        print()

        print("-------- Dataset Haberman  --------\n")
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
