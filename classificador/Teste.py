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

    balanceScaleFile = os.path.join(os.getcwd(), "balanceScale_output.txt")
    balanceScaleWriter = codecs.open(balanceScaleFile, 'w', "utf-8")

    glassFile = os.path.join(os.getcwd(), "glass_output.txt")
    glassWriter = codecs.open(glassFile, 'w', "utf-8")

    habermanFile = os.path.join(os.getcwd(), "haberman_output.txt")
    habermanWriter = codecs.open(habermanFile, 'w', "utf-8")

    imgSegFile = os.path.join(os.getcwd(), "imgSegment_output.txt")
    imgSegWriter = codecs.open(imgSegFile, 'w', "utf-8")

    ionosphereFile = os.path.join(os.getcwd(), "ionosphere_output.txt")
    ionosphereWriter = codecs.open(ionosphereFile, 'w', "utf-8")

    irisFile = os.path.join(os.getcwd(), "iris_output.txt")
    irisWriter = codecs.open(irisFile, 'w', "utf-8")

    primaIndiansFile = os.path.join(os.getcwd(), "primaIndians_output.txt")
    primaIndiansWriter = codecs.open(primaIndiansFile, 'w', "utf-8")

    sonarAllFile = os.path.join(os.getcwd(), "sonarAll_output.txt")
    sonarAllWriter = codecs.open(sonarAllFile, 'w', "utf-8")

    spamBaseFile = os.path.join(os.getcwd(), "spamBase_output.txt")
    spamBaseWriter = codecs.open(spamBaseFile, 'w', "utf-8")

    wineFile = os.path.join(os.getcwd(), "wine_output.txt")
    wineWriter = codecs.open(wineFile, 'w', "utf-8")


    if __name__ == "__main__":

        irisWriter.write("-------- Dataset Iris --------")
        print("-------- Dataset Iris --------\n")
        IrisClassifier().printAccruraciesAndStds(irisWriter)
        print()

        primaIndiansWriter.write("-------- Dataset Prima Indians Diabetes --------")
        print("-------- Dataset Prima Indians Diabetes --------\n")
        DiabetesClassifier().printAccruraciesAndStds(primaIndiansWriter)
        print()

        balanceScaleWriter.write("-------- Dataset Balance Scale --------")
        print("-------- Dataset Balance Scale --------\n")
        BalanceScaleClassifier().printAccruraciesAndStds(balanceScaleWriter)
        print()

        sonarAllWriter.write("-------- Dataset Sonar All --------")
        print("-------- Dataset Sonar All --------\n")
        SonarClassifier().printAccruraciesAndStds(sonarAllWriter)
        print()

        glassWriter.write("-------- Dataset Glass  --------")
        print("-------- Dataset Glass  --------\n")
        GlassClassifier().printAccruraciesAndStds(glassWriter)
        print()

        habermanWriter.write("-------- Dataset Haberman  --------")
        print("-------- Dataset Haberman  --------\n")
        HabermanClassifier().printAccruraciesAndStds(habermanWriter)
        print()

        ionosphereWriter.write("-------- Dataset Ionsphere --------")
        print("-------- Dataset Ionsphere --------\n")
        IonosphereClassifier().printAccruraciesAndStds(ionosphereWriter)
        print()

        spamBaseWriter.write("-------- Dataset Spams --------")
        print("-------- Dataset Spams --------\n")
        SpamClassifier().printAccruraciesAndStds(spamBaseWriter)
        print()

        imgSegWriter.write("-------- Dataset Image Segmentation --------")
        print("-------- Dataset Image Segmentation --------\n")
        ImageSegmentationClassifier().printAccruraciesAndStds(imgSegWriter)
        print()

        wineWriter.write("-------- Dataset Wine --------")
        print("-------- Dataset Wine --------\n")
        WineClassifier().printAccruraciesAndStds(wineWriter)
        print()
