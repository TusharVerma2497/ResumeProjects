from ..lib.colorTransfer import applyColorTransfer
from ..lib.gui import get_swatches
from ..lib.sampling import swatchSampling
from ..lib.luminance import transferColor
import numpy as np
import random
from PIL import Image
from ...Part1.lib.helper import  showImage, RGBtoLAB, GRAYtoLAB, readImage, LABtoRGB, calculate_mode, saveImage, binary_search


def applyWithinSwatchesColorTransfer(swatches_list, sourceLAB, targetL, neighbourSDKernelSize, alpha, applyNeighbourSD=True, samplingSize=50):
    coloredSwatches=[]
    for i in range(0,len(swatches_list[0]),2):
        #swatch from source image
        a,c=swatches_list[0][i]
        b,d=swatches_list[0][i+1]

        # corresponding swatch from target image
        e,g=swatches_list[1][i]
        f,h=swatches_list[1][i+1]

        colorTransfferedSwatch=applyColorTransfer(sourceLAB[a:b,c:d,:], targetL[e:f,g:h], samplingSize, neighbourSDKernelSize, alpha, applyNeighbourSD)
        coloredSwatches.append(colorTransfferedSwatch)
    return coloredSwatches


def getNeighbourhoodForN_RandomPoints(list, n=6, kernelSize=5):
    neighbourList=[]
    for swatch in list:
        originalSwatchShape=swatch.shape
        #padding the image to apply filter of odd length
        pad_width=kernelSize//2
        swatch=np.pad(swatch, pad_width=((pad_width,pad_width),(pad_width,pad_width),(0,0)), mode='edge',)
        neighbours=np.zeros((kernelSize,kernelSize,n))
        for i in range(n):
            # index=(random.randint(0,originalSwatchShape[0]-1), random.randint(0,originalSwatchShape[1]-1))
            index=(random.randint(0,originalSwatchShape[0]-5), random.randint(0,originalSwatchShape[1]-5))
            neighbours[:,:,i]=swatch[index[0]:index[0]+kernelSize, index[1]:index[1]+kernelSize, 0]
        neighbourList.append(neighbours)
    return neighbourList



def globalColorTransferUsingSwatches(source, target, neighbourSDKernelSize, fast=True, alpha=0.6, kernelSize=5, numberOfNeighbour=16,saveColoredSwatches=False):
    # if fast is true kernel level color filling will happen else pixel level
    if(fast):
        increment=kernelSize
    else:
        increment=1
    
    #convertin the source and target images to Lab color space
    sourceLAB=RGBtoLAB(source)
    targetL=GRAYtoLAB(target)

    #using a gui application to select swatches from images
    swatches_list=get_swatches(Image.fromarray(source), Image.fromarray(targetL))

    #applying the color transfer among corresponding swatches
    RGBSwatchList=applyWithinSwatchesColorTransfer(swatches_list, sourceLAB, targetL, neighbourSDKernelSize, alpha, samplingSize=250, applyNeighbourSD=False)
    
    #saving the colored swatch if required
    if(saveColoredSwatches):
        for i in range(len(RGBSwatchList)):
            saveImage(RGBSwatchList[i], name=f'Colored_swatch_{i}_',part=3)

    #converting swatches to LAB color space
    LABSwatchList=[RGBtoLAB(i) for i in RGBSwatchList]

    #collecting n samples from colored swatches
    samplesFromColoredSwatches=swatchSampling(LABSwatchList,sampleSize=50)

    #saving n neighbours from each swatch randomly for tecture computation
    neighbours=getNeighbourhoodForN_RandomPoints(LABSwatchList,kernelSize=kernelSize,n=numberOfNeighbour)

    # #padding the target image with edge values to ease the neighbour calculation of corner pixels
    finalImage=np.zeros((targetL.shape[0]+kernelSize-1 ,targetL.shape[1]+kernelSize-1,3),dtype=np.uint8)
    pad_width=kernelSize//2
    targetL=np.pad(targetL, pad_width=pad_width, mode='edge')
    #compint the luminance values in the final image as it is
    finalImage[:,:,0]=targetL
    
    #finally coloring each pixel with best swatch by comparing it's neighbourhood with swatches'
    for i in range(0,targetL.shape[0]-kernelSize,increment):
        for j in range(0,targetL.shape[1]-kernelSize,increment):

            #getting the neighbourhood of the to be colored pixel
            temp1=targetL.reshape(targetL.shape[0],targetL.shape[1],1)
            temp=temp1[i:i+kernelSize,j:j+kernelSize,:]

            # temp will be used to comparing with neighbors of a particular swatch all at once 
            temp=np.repeat(temp,numberOfNeighbour,-1)

            # each swatch comparision will be saved in s for bwtween swatch comparisions
            s=np.zeros((len(neighbours),numberOfNeighbour))

            for k in range(len(neighbours)):
                s[k]=np.sum((temp-neighbours[k])**2, axis=(0,1))
            
            #calculating the least dissimilar swatch with the target neighbour
            mostSimilarSwatch=np.argmin(np.sum(s,axis=1))

            #coloring the pixel
            if(fast):
                finalImage[i:i+kernelSize,j:j+kernelSize,:] = transferColor(samplesFromColoredSwatches[mostSimilarSwatch], targetL[i:i+kernelSize,j:j+kernelSize], targetL[i:i+kernelSize,j:j+kernelSize])
            else:
                finalImage[i,j][1:]=samplesFromColoredSwatches[mostSimilarSwatch][binary_search(targetL[i,j], samplesFromColoredSwatches[mostSimilarSwatch])][1:]
            

    #removing padding
    finalImage=finalImage[:finalImage.shape[0]-kernelSize,:finalImage.shape[1]-kernelSize,:]

    # returning the RGB colored target
    return LABtoRGB(finalImage)
    # finalImage[:,:,0]*=0
    # finalImage[:,:,0]+=100
    # showImage(LABtoRGB(finalImage))