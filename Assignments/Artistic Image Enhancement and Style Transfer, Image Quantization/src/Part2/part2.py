from ..Part1.lib.helper import showImage, readImage, saveImage, changeRange
import numpy as np
import sys
from ..Part2.lib.medianCut import quantize
from ..Part2.lib.floydSteinbergDithering import applyDithering
from ..Part1.part1 import artisticEnhancement


def quantizedRendering(img_path, quantizationColorCount=8,  saveImages=False):

    # getting artistic enhanced image from part1
    artisticImage=artisticEnhancement(img_path)

    #performaing quantization of image
    quantizedImage=quantize(artisticImage,numOfColors=quantizationColorCount)

    #performing dithering to the quantized image
    ditheredImage=applyDithering(artisticImage/255, quantizedImage/255)

    if(saveImages):
        saveImage(changeRange(artisticImage,0,255).astype(np.uint8), name='Artistic_img_',part=2)
        saveImage(changeRange(quantizedImage,0,255).astype(np.uint8), name='Quantized_img_',part=2)
        saveImage(changeRange(ditheredImage,0,255).astype(np.uint8), name='Dithered_img_',part=2)

    return quantizedImage, ditheredImage




if __name__ == "__main__":
    quantizationColorCount=12
    if(len(sys.argv)==3):
        quantizationColorCount=int(sys.argv[2])

    #invoking part2
    quantizedImage, ditheredImage =quantizedRendering(img_path=sys.argv[1], quantizationColorCount=quantizationColorCount, saveImages=True)
    showImage(quantizedImage, f'Quantized Image {quantizationColorCount} Colors')
    showImage(ditheredImage, f'Dithered Image')
