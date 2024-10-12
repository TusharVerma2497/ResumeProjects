import numpy as np
from .helper import BGRtoLAB, LABtoRGB, readImageCV2


def getChromaticMap(path):
    image=readImageCV2(path)
    LAB=BGRtoLAB(image)
    #defining a constant illumination setting the illumination to mean of the L channel
    temp=np.ones(shape=LAB.shape[0:2])*np.mean(LAB[:,:,0])
    LAB[:,:,0]=temp
    # LAB[:,:,1]=changeRange(LAB[:,:,1],-255,255)
    # LAB[:,:,2]=changeRange(LAB[:,:,2],-255,255)
    RGB=LABtoRGB(LAB)
    return LAB



def combineShadowAndCMapImages(SI, CM, ro):
    combinedImage=SI*(1+np.tanh(ro*(CM-256)))/2
    # print(combinedImage)
    return combinedImage



def artisticRendering(LD,image,Beta):
    LD=np.repeat(LD.reshape(LD.shape[0], LD.shape[1],1),3,2)
    return np.where(LD == 0, Beta*image + (1-Beta)*LD, image)