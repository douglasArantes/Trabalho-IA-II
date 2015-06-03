import os
import codecs

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

    outputFile = os.path.join(os.getcwd(), "output.txt")
    fileWriter = codecs.open(outputFile, 'w', "utf-8")


    if __name__ == "__main__":

        fileWriter.write("-------- Dataset Iris --------")
        print("-------- Dataset Iris --------\n")
        IrisClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Prima Indians Diabetes --------")
        print("-------- Dataset Prima Indians Diabetes --------\n")
        DiabetesClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Balance Scale --------")
        print("-------- Dataset Balance Scale --------\n")
        BalanceScaleClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Sonar All --------")
        print("-------- Dataset Sonar All --------\n")
        SonarClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Glass  --------")
        print("-------- Dataset Glass  --------\n")
        GlassClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Haberman  --------")
        print("-------- Dataset Haberman  --------\n")
        HabermanClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Ionsphere --------")
        print("-------- Dataset Ionsphere --------\n")
        IonosphereClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Spams --------")
        print("-------- Dataset Spams --------\n")
        SpamClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Image Segmentation --------")
        print("-------- Dataset Image Segmentation --------\n")
        ImageSegmentationClassifier().printAccruraciesAndStds(fileWriter)
        print()

        fileWriter.write("-------- Dataset Wine --------")
        print("-------- Dataset Wine --------\n")
        WineClassifier().printAccruraciesAndStds(fileWriter)
        print()
