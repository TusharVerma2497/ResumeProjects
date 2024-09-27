from gui import get_swatches
from PIL import Image
import sys
import numpy as np
from helper import readImage
from warping import reverseWarpingUsingBarycentricCoordinate
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

    os.makedirs("task2_results", exist_ok=True)

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
        arr=np.load(filePath+'.npy')
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

    #performing inverse warping form I2 to I1
    I2_=reverseWarpingUsingBarycentricCoordinate(I2,I1,I2Points,I1Points)
    #performing inverse warping form I2_ to I3
    I4=reverseWarpingUsingBarycentricCoordinate(I2_,I3,I1Points,I2Points)


    #saving
    TransformedImage = Image.fromarray(I2_)
    TransformedImage.save(os.path.join('task2_results',f"I2'_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))
    TransformedImage = Image.fromarray(I4)
    TransformedImage.save(os.path.join('task2_results',f"I4'_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))