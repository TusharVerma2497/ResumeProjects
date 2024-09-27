import numpy as np
from ...Part1.lib.helper import binary_search


def getLuminanceMeanAndSD(imageGray):
    return np.mean(imageGray), np.std(imageGray)


def neighbourSD(image,kernelSize=5):

    h,w = image.shape
    #creating a numpy array for neighbour SD
    std=np.empty_like(image)

    #padding the image to apply filter of odd length
    pad_width=kernelSize//2
    image=np.pad(image, pad_width=pad_width, mode='edge')

    for i in range(h):
        for j in range(w):
            std[i,j]=np.std(image[i:i+kernelSize, j:j+kernelSize])
    return std


def transferColor(samples, scoreMap, targetL):
    h,w = scoreMap.shape
    finalImage=np.zeros((h,w,3),dtype=np.uint8)
    finalImage[:,:,0]=targetL

    #applying binary search to find the nearest sample to a given score in the scoreMap
    for i in range(h):
        for j in range(w):
            pixel=scoreMap[i,j]
            pos=binary_search(pixel,samples)
            finalImage[i,j,1:]=samples[pos][1:]
    # showImage(finalImage)
    return finalImage





