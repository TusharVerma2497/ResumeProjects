from gui import get_swatches
from PIL import Image
import sys
import numpy as np
from helper import readImage, getAffineTransformUsingCV2, performTransformation
from warping import reverseWarpingUsingBarycentricCoordinate, findBoundaryOfWarpedImage, reverseWarping
from delaunayTriangulation import bowyer_watson, plot_delaunay, Triangle, mapTriangles
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

    os.makedirs("task3_results", exist_ok=True)

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

        #adding the cornert points to the set of anchor points
        I1Corners=[(0,0),(I1.shape[0],I1.shape[1]),(0,I1.shape[1]),(I1.shape[0],0)]
        I2Corners=[(0,0),(I2.shape[0],I2.shape[1]),(0,I2.shape[1]),(I2.shape[0],0)]
        I3Corners=[(0,0),(I3.shape[0],I3.shape[1]),(0,I3.shape[1]),(I3.shape[0],0)]
        I1Points+=I1Corners
        I2Points+=I2Corners
        I3Points+=I3Corners
        n=n+4

        I1Points = np.float32(I1Points)
        I2Points = np.float32(I2Points)
        I3Points = np.float32(I3Points)
        if(len(I1Points)==n and len(I2Points)==n and len(I3Points)==n):
            arr=np.array([I1Points, I2Points, I3Points])
            np.save(filePath, arr)
    
    print(f'name of the file containing the saved anchors:\n{hash_hex}')

    # performing delaunay triangulations
    trianglesA=bowyer_watson(I1Points)
    plot_delaunay(trianglesA, os.path.join("task3_results",f"I1Triangles_{nameI1}_{nameI2}_{nameI3}_{n}"))
    
    # triangulation will not be performed on the other two images 
    # because we will simply nap the corresponding anchor points to get the triangles for the other images
    trianglesB=mapTriangles(copy.deepcopy(trianglesA), I1Points, I2Points)
    plot_delaunay(trianglesB, os.path.join("task3_results",f"I2Triangles_{nameI1}_{nameI2}_{nameI3}_{n}"))
    
    trianglesC=mapTriangles(copy.deepcopy(trianglesA), I1Points, I3Points)
    plot_delaunay(trianglesC, os.path.join("task3_results",f"I3Triangles_{nameI1}_{nameI2}_{nameI3}_{n}"))


    # Using barycentric coordinates
    I2_=copy.deepcopy(I1)
    for i in range(len(trianglesA)):
        t1=trianglesA[i]
        t2=trianglesB[i]
        t1=np.array([[t1.p1.x,t1.p1.y],[t1.p2.x,t1.p2.y],[t1.p3.x,t1.p3.y]])
        t2=np.array([[t2.p1.x,t2.p1.y],[t2.p2.x,t2.p2.y],[t2.p3.x,t2.p3.y]])
        #performing warp
        I2_=reverseWarpingUsingBarycentricCoordinate(I2,I2_,t2,t1,carry=True)
    #saving
    TransformedImage = Image.fromarray(I2_)
    TransformedImage.save(os.path.join('task3_results',f"barycentric_I2'_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

    I4=copy.deepcopy(I3)
    for i in range(len(trianglesA)):
        t3=trianglesC[i]
        t1=trianglesA[i]
        t3=np.array([[t3.p1.x,t3.p1.y],[t3.p2.x,t3.p2.y],[t3.p3.x,t3.p3.y]])
        t1=np.array([[t1.p1.x,t1.p1.y],[t1.p2.x,t1.p2.y],[t1.p3.x,t1.p3.y]])
        #performing warp
        I4=reverseWarpingUsingBarycentricCoordinate(I2_,I4,t1,t3,carry=True)
    #saving
    TransformedImage = Image.fromarray(I4)
    TransformedImage.save(os.path.join('task3_results',f"barycentric_I4_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

    print('Results based on Barycentric Coordinates has been saved.')


    # Using affine warp matrix
    I1_=np.zeros(I2.shape,dtype=np.uint8)
    I3_=np.zeros(I2.shape,dtype=np.uint8)
    I4=np.zeros(I3.shape,dtype=np.uint8)
    for i in range(len(trianglesA)):
        t1=trianglesA[i]
        t2=trianglesB[i]
        t3=trianglesC[i]
        t1=np.array([[t1.p1.x,t1.p1.y],[t1.p2.x,t1.p2.y],[t1.p3.x,t1.p3.y]])
        t2=np.array([[t2.p1.x,t2.p1.y],[t2.p2.x,t2.p2.y],[t2.p3.x,t2.p3.y]])
        t3=np.array([[t3.p1.x,t3.p1.y],[t3.p2.x,t3.p2.y],[t3.p3.x,t3.p3.y]])

        #finding warp matrix H1
        warpMatrixH1=getAffineTransformUsingCV2(t1,t2)

        # transforming the triangle coordinates of I1 and I3 through the warp matrix estimated
        t1_=performTransformation(t1, warpMatrixH1)
        t3_=performTransformation(t3, warpMatrixH1)

        # finding warp matrix between I1'and I3' for final transformation of I2 to I4 
        warpMatrixH3=getAffineTransformUsingCV2(t1,t3)
        
        #inverse warping to find I1', I3' and I4, for faster processing we may direcly find I4 since warp matrixH3 has already been found
        I1_=reverseWarping(warpMatrixH1, I1, I1_, t1)
        I3_=reverseWarping(warpMatrixH1, I3, I3_, t3)
        I4=reverseWarping(warpMatrixH3, I2, I4, t3)
    
    #saving
    TransformedImage = Image.fromarray(I1_)
    TransformedImage.save(os.path.join('task3_results',f"warpMatrix_I1'_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

    TransformedImage = Image.fromarray(I3_)
    TransformedImage.save(os.path.join('task3_results',f"warpMatrix_I3'_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

    TransformedImage = Image.fromarray(I4)
    TransformedImage.save(os.path.join('task3_results',f"warpMatrix_I4_{nameI1}_{nameI2}_{nameI3}_{n}.jpg"))

    print('Results based on warp matrix has been saved.')