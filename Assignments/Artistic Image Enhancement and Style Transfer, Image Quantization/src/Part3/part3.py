import sys
from ..Part1.lib.helper import  showImage, RGBtoLAB, GRAYtoLAB, readImage, saveImage
from ..Part3.lib.swatches import globalColorTransferUsingSwatches
from ..Part3.lib.colorTransfer import applyColorTransfer
from ..Part1.part1 import artisticEnhancement

# hyperparameters
samplingSize=40
applyNeighbourSD=False
neighbourSDKernelSize=5
alpha=0.6
kernelSize=7
numberOfNeighbour=100
fast=True,


if __name__ == "__main__":
    # reading source(source of color RGB) and target(GRAY) image
    # getting artistic enhanced image from part1
    source=artisticEnhancement(sys.argv[1])
    target=readImage(sys.argv[2])

    # invoking part 3.1: COLOR TRANSFER
    #convertin the source and target images to Lab color space
    sourceLAB=RGBtoLAB(source)
    targetL=GRAYtoLAB(target)
    colorTransfferedImage=applyColorTransfer(
        sourceLAB,
        targetL,
        samplingSize,
        neighbourSDKernelSize,
        alpha,
        applyNeighbourSD=applyNeighbourSD)
    saveImage(colorTransfferedImage, name='Basic_Color_transfer_',part=3)
    print('saved image for basic color transfer')


    # invoking part 3.2: SWATCHES
    finalImage=globalColorTransferUsingSwatches(
        source,
        target,
        neighbourSDKernelSize,
        fast,
        alpha,
        kernelSize,
        numberOfNeighbour,
        saveColoredSwatches=True)
    saveImage(finalImage, name='Swatch_Color_transfer_',part=3)
    print('saved image for swatch based color transfer')