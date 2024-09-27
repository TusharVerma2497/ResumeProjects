
from gui2 import get_swatches
from PIL import Image
import sys
import numpy as np
from helper import readImage, RGBtoLAB
from warping import reverseWarpingUsingBarycentricCoordinate
from delaunayTriangulation import bowyer_watson, mapTriangles, plot_delaunay
import hashlib
import os
import copy
from colorTransfer import getLuminanceMeanAndSD
import cv2
from smoothing import getMask
from pathlib import Path

if __name__ == "__main__":
    I1=readImage(sys.argv[1])
    I2=readImage(sys.argv[2])
    n=int(sys.argv[3])

    nameI1 = Path(sys.argv[1]).name[:-4]
    nameI2 = Path(sys.argv[2]).name[:-4]
    nameI3 = Path(sys.argv[3]).name[:-4]

    os.makedirs("part2_results", exist_ok=True)

     # Create a string to be hashed
    my_string=' '.join(sys.argv[1:])
    # Create a hash object using SHA-256
    hash_object = hashlib.sha256()
    # Update the hash object with the bytes representation of the string
    hash_object.update(my_string.encode())
    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()

    #caching
    filePath=os.path.join('cache',hash_hex)
    if os.path.exists(filePath+".npy"):
        arr=np.load(filePath+".npy")
        I1Points=arr[0]
        I2Points=arr[1]
    else:
        #gui to get anchor points incase they aren't cached
        I1Points, I2Points = get_swatches(Image.fromarray(I1), Image.fromarray(I2), n)
        I1Points = np.float32(I1Points)
        I2Points = np.float32(I2Points)
        if(len(I1Points)==n and len(I2Points)==n):
            arr=np.array([I1Points, I2Points])
            np.save(filePath, arr)

    print(f'name of the file containing the saved anchors:\n{hash_hex}')

    # performing delaunay triangulations
    trianglesA=bowyer_watson(I1Points)
    # triangulation will not be performed on the target image
    # because we will simply nap the corresponding anchor points to get the triangles for the other image
    trianglesB=mapTriangles(copy.deepcopy(trianglesA), I1Points, I2Points)

    #performing Z mormalization on the Luminance part of the source pixel 
    source_mean, source_std = getLuminanceMeanAndSD(RGBtoLAB(I1)[:,:,0])
    target_mean, target_std = getLuminanceMeanAndSD(RGBtoLAB(I2)[:,:,0])
    I1_=copy.deepcopy(RGBtoLAB(I1))
    I2_=copy.deepcopy(I2)
    #Normalizing
    I1_[:,:,0]= target_std/source_std * (I1[:,:,0]-source_mean) + target_mean

    

# global allignment using delaunay triangles
    mask= np.zeros(shape=I2.shape, dtype=np.float32)
    for i in range(len(trianglesA)):
        t1=trianglesA[i]
        t2=trianglesB[i]
        t1=np.array([[t1.p1.x,t1.p1.y],[t1.p2.x,t1.p2.y],[t1.p3.x,t1.p3.y]])
        t2=np.array([[t2.p1.x,t2.p1.y],[t2.p2.x,t2.p2.y],[t2.p3.x,t2.p3.y]])
        #performing inverse warping to copy source Luminance part of the pixel to the target luminance part of the pixel
        I2=reverseWarpingUsingBarycentricCoordinate(I1_,RGBtoLAB(I2),t1,t2,name='swappedFace1', carry=True, colorTransfer=True)
        #also calulating mask to smoothing in later portion
        mask=getMask(mask,t2)
    
    TransformedImage = Image.fromarray(mask.astype(np.uint8))
    TransformedImage.save(os.path.join('part2_results',f"mask_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

    #performing gaussian blur on the mask
    mask = cv2.GaussianBlur(mask, (151, 151), 0)

    TransformedImage = Image.fromarray(mask.astype(np.uint8))
    TransformedImage.save(os.path.join('part2_results',f"gaussianMask_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

    #normalizing the mask array
    mask=mask/255

    #blending and smoothing the final image
    result = I2_ * (1 - mask) + I2 * mask
    result=result.astype(np.uint8)

    #saving image
    TransformedImage = Image.fromarray(I2)
    TransformedImage.save(os.path.join('part2_results',f"colorAdjusted_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))
    TransformedImage = Image.fromarray(result)
    TransformedImage.save(os.path.join('part2_results',f"borderSmoothend_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

