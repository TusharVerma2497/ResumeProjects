from gui import get_swatches
from PIL import Image
import sys
import numpy as np
from helper import readImage, getAffineTransformSolvingSLI, getAffineTransformUsingCV2, performTransformation
from warping import reverseWarping
import hashlib
import os
import copy
from pathlib import Path

if __name__ == "__main__":
    I1=readImage(sys.argv[1])
    I2=readImage(sys.argv[2])
    I3=readImage(sys.argv[3])
    n=int(sys.argv[4])
    

    nameI1 = Path(sys.argv[1]).name[:-4]
    nameI2 = Path(sys.argv[2]).name[:-4]
    nameI3 = Path(sys.argv[3]).name[:-4]

    os.makedirs("task1_results", exist_ok=True)
     # Create a string to be hashed
    my_string=' '.join(sys.argv[1:])
    # Create a hash object using SHA-256
    hash_object = hashlib.sha256()
    # Update the hash object with the bytes representation of the string
    hash_object.update(my_string.encode())
    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()

    
    #caching the anchor points
    filePath=os.path.join('cache',hash_hex)
    if os.path.exists(filePath+".npy"):
        arr=np.load(filePath+".npy")
        I1Points=arr[0]
        I2Points=arr[1]
        I3Points=arr[2]
    else:
        #gui to get anchor points incase they aren't cached
        I1Points, I2Points, I3Points = get_swatches(Image.fromarray(I1), Image.fromarray(I2), Image.fromarray(I3), n)
        I1Points = np.float32(I1Points)
        I2Points = np.float32(I2Points)
        I3Points = np.float32(I3Points)
        if(len(I1Points)==n and len(I2Points)==n and len(I3Points)==n):
            arr=np.array([I1Points, I2Points, I3Points])
            np.save(filePath, arr)

    print(f'name of the file containing the saved anchors:\n{hash_hex}')
    # estimating the warp matrix H1
    warpMatrixH1=getAffineTransformUsingCV2(I1Points,I2Points)

    
    I1_=np.zeros(I2.shape, dtype=np.uint8)
    I1_=reverseWarping(warpMatrixH1, I1, I1_, I1Points)

   
    I2_=np.zeros(I2.shape, dtype=np.uint8)
    I2_=reverseWarping(warpMatrixH1, I3, I2_, I3Points)

    #saving
    TransformedImage = Image.fromarray(I1_)
    TransformedImage.save(os.path.join('task1_results',f"I1'_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))
    TransformedImage = Image.fromarray(I2_)
    TransformedImage.save(os.path.join('task1_results',f"I3'_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

    # transforming the triangle coordinates of I1 and I3 through the warp matrix estimated
    I1_=performTransformation(I1Points, warpMatrixH1)
    I3_=performTransformation(I3Points, warpMatrixH1)

    #estimating the warp matrix H2
    warpMatrixH2=getAffineTransformUsingCV2(I1_,I3_)

   
    I4=np.zeros(I3.shape, dtype=np.uint8)
    I4=reverseWarping(warpMatrixH2, I2, I4, I2Points)

    #saving
    TransformedImage = Image.fromarray(I4)
    TransformedImage.save(os.path.join('task1_results',f"I4_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))