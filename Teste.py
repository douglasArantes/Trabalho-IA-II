import os
import codecs

from IrisClassifier import IrisClassifier
from DiabetesClassifier import DiabetesClassifier
from BalanceScaleClassifier import BalanceScaleClassifier
from BanknoteAuthenticationClassifier import BanknoteAuthenticationClassifier
from GlassClassifier import GlassClassifier
from HabermanClassifier import HabermanClassifier
from IonosphereClassifier import IonosphereClassifier
from SpamClassifier import SpamClassifier
from ImageSegmentationClassifier import ImageSegmentationClassifier
from WineClassifier import WineClassifier


class Teste:

    balanceScaleFile = os.path.join(os.getcwd(), "outputs/balanceScale_output.txt")
    balanceScaleWriter = codecs.open(balanceScaleFile, 'w', "utf-8")

    glassFile = os.path.join(os.getcwd(), "outputs/glass_output.txt")
    glassWriter = codecs.open(glassFile, 'w', "utf-8")

    habermanFile = os.path.join(os.getcwd(), "outputs/outphaberman_output.txt")
    habermanWriter = codecs.open(habermanFile, 'w', "utf-8")

    imgSegFile = os.path.join(os.getcwd(), "outputs/imgSegment_output.txt")
    imgSegWriter = codecs.open(imgSegFile, 'w', "utf-8")

    ionosphereFile = os.path.join(os.getcwd(), "outputs/ionosphere_output.txt")
    ionosphereWriter = codecs.open(ionosphereFile, 'w', "utf-8")

    irisFile = os.path.join(os.getcwd(), "outputs/iris_output.txt")
    irisWriter = codecs.open(irisFile, 'w', "utf-8")

    primaIndiansFile = os.path.join(os.getcwd(), "outputs/primaIndians_output.txt")
    primaIndiansWriter = codecs.open(primaIndiansFile, 'w', "utf-8")

    banknoteFile = os.path.join(os.getcwd(), "outputs/banknoteAuthentication_output.txt")
    banknoteWriter = codecs.open(banknoteFile, 'w', "utf-8")

    spamBaseFile = os.path.join(os.getcwd(), "outputs/spamBase_output.txt")
    spamBaseWriter = codecs.open(spamBaseFile, 'w', "utf-8")

    wineFile = os.path.join(os.getcwd(), "outputs/wine_output.txt")
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

        banknoteWriter.write("-------- Dataset Banknote --------")
        print("-------- Dataset Banknote --------\n")
        BanknoteAuthenticationClassifier().printAccruraciesAndStds(banknoteWriter)
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
