import math
from random import randint
import numpy as np
from helper import pointInBoundary, LABtoRGB, showImage
from PIL import Image

def jitteredSampling(image, boundary, sampleSize=200, gridDivisionFactor=10, name='out'):
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
                    if(pointInBoundary(i+randomOffsetX,j+randomOffsetY,boundary)):
                        sampleList.append(image[i+randomOffsetX, j+randomOffsetY])
    #                     image[i+randomOffsetX, j+randomOffsetY]=np.array([0,0,0],dtype=np.uint8)
    # TransformedImage = Image.fromarray(image)
    # TransformedImage.save(name+".jpg")
    return sampleList



def binary_search(target,samples):
    target=float(target[0])
    left, right = 0, len(samples) - 1
    leftClosest=left
    rightClosest=right
    
    while left < right:
        mid = left + (right - left) // 2

        if samples[mid][0] < target:
            left = mid + 1
            leftClosest=left
        else:
            right = mid - 1
            rightClosest=right

    #if perfect match not found the returning the closest one of the leftClosest and rightClosest
    return leftClosest if target-samples[leftClosest][0] <= samples[rightClosest][0]-target else rightClosest




def getLuminanceMeanAndSD(imageGray):
    return np.mean(imageGray), np.std(imageGray)