from ...Part1.lib.helper import LABtoRGB 
from ..lib.sampling import jitteredSampling
from ..lib.luminance import getLuminanceMeanAndSD, neighbourSD, transferColor

def applyColorTransfer(sourceLAB,targetL, samplingSize, neighbourSDKernelSize, alpha, applyNeighbourSD=True):

    #getting jittered samples from image pixels are sorted in ascending order on the basis of intensity
    samples=jitteredSampling(sourceLAB, gridDivisionFactor=5, sampleSize=samplingSize)

    #calculating mean and variance of source and target image for luminance remapping
    source_mean, source_std = getLuminanceMeanAndSD(sourceLAB[:,:,0])
    target_mean, target_std = getLuminanceMeanAndSD(targetL) 


    #performing target luminance remapping
    # targetL= source_std/target_std * (targetL-target_mean) + source_mean
    sourceLAB[:,:,0]= target_std/source_std * (sourceLAB[:,:,0]-source_mean) + target_mean

    if(applyNeighbourSD):
        # calculating the standard deviation of neighbouring pixel
        targetNeighbourSD=neighbourSD(targetL,kernelSize=neighbourSDKernelSize)

        # finding the weighted average of luminance (50%) and standard deviation (50%)
        scoreMap = targetL*alpha + (1-alpha)*targetNeighbourSD
        # print(targetL.shape)
    else:
         scoreMap = targetL

    finalImageLAB=transferColor(samples, scoreMap, targetL)
    # finalImageLAB=transferColor(samples, scoreMap)

    finalImageRGB=LABtoRGB(finalImageLAB)

    return finalImageRGB