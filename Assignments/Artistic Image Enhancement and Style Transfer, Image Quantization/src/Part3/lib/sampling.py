import math
from random import randint
import numpy as np

def jitteredSampling(image, sampleSize=200, gridDivisionFactor=10):
    sampleList=[]
    h,w=image.shape[:2]
    # print(image.shape)
    h_len_grid=math.floor(h/gridDivisionFactor)
    w_len_grid=math.floor(w/gridDivisionFactor)
    # print(h_len_grid,w_len_grid)
    samplingPerGrid=math.floor(sampleSize/gridDivisionFactor**2)
    while(len(sampleList)<sampleSize):
        for i in range(0, h-h_len_grid, h_len_grid):
            for j in range(0, w-w_len_grid,w_len_grid):
                for k in range(samplingPerGrid):
                    randomOffsetX=randint(0,h_len_grid-1)
                    randomOffsetY=randint(0,w_len_grid-1)
                    sampleList.append(image[i+randomOffsetX, j+randomOffsetY])

    # getting first sampleSize samples and sorting them according to luminance
    sampleList=np.array(sampleList[:sampleSize])
    sorted_indices = np.argsort(sampleList[:,0])
    return sampleList[sorted_indices]




def swatchSampling(list, sampleSize):
    sampleList=[]
    for swatch in list:
        sampleList.append(jitteredSampling(swatch, gridDivisionFactor=2, sampleSize=sampleSize))
    return sampleList