import numpy as np
from helper import performTransformation, getBarycentricCoordinates, pointInBoundary, getPointUsingBarycentric, areaTriangle
from PIL import Image
import os

def findBoundaryOfWarpedImage(image,warpMatrix):
    #finding the border mapping
    boundaryPoints=np.float32([[0,0],[0,image.shape[1]],[image.shape[0],0],[image.shape[0],image.shape[1]]])
    transformedBoundaryPoints=performTransformation(boundaryPoints,warpMatrix)
    # print(transformedBoundaryPoints)
    #finding the min and max of the the transformed axis
    min=np.min(transformedBoundaryPoints, axis=0)
    max=np.max(transformedBoundaryPoints, axis=0)
    minx=int(np.ceil(min[0]))
    miny=int(np.ceil(min[1]))
    maxx=int(np.ceil(max[0]))
    maxy=int(np.ceil(max[1]))
    return minx, miny, maxx, maxy




def reverseWarping(warpMatrix, source, target, sourceBoundary=[], name='output'):
    
    inverse_warpMatrix = np.linalg.inv(np.vstack((warpMatrix,np.array([0,0,1]))))
    
    if len(sourceBoundary)==3:
        # minBoundary=(minx, miny)
        targetBoundary=performTransformation(sourceBoundary,warpMatrix)
        #finding the min and max of the the transformed axis
        min=np.min(targetBoundary, axis=0)
        max=np.max(targetBoundary, axis=0)
        minx=int(np.round(min[0]))
        miny=int(np.round(min[1]))
        maxx=int(np.round(max[0]))
        maxy=int(np.round(max[1]))
        # print(transformedPoints)

        A = np.round(areaTriangle(targetBoundary[0][0], targetBoundary[0][1],
                      targetBoundary[1][0], targetBoundary[1][1],
                      targetBoundary[2][0], targetBoundary[2][1]))
        pointsInsideTargetTriangle=[]
        
        
        for i in range(minx,maxx,1):
            for j in range(miny, maxy,1):
                # print(i,j)
                if(pointInBoundary(i,j,targetBoundary,A)):
                    pointsInsideTargetTriangle.append([i,j])

        
        pointsInsideTargetTriangle=np.float32(pointsInsideTargetTriangle)
        pointsInsideSourceTriangle=performTransformation(pointsInsideTargetTriangle,inverse_warpMatrix)

        sourceColors=[]
        for i in pointsInsideSourceTriangle:
            x=int(i[0])
            y=int(i[1])
            if(x<0 or y<0 or x>=source.shape[0] or y>=source.shape[1]):
                sourceColors.append(np.uint8([0,0,0]))
            else:
                sourceColors.append(source[x,y])


        for i in range(len(pointsInsideTargetTriangle)):
            x=int(pointsInsideTargetTriangle[i,0])
            y=int(pointsInsideTargetTriangle[i,1])
            # print(x,y)
            target[x,y]=sourceColors[i]
            # finalImage[x,y]=np.uint8([0,0,0])
        # TransformedImage = Image.fromarray(finalImage)
        # TransformedImage.save(os.path.join('task1_results',name+".jpg"))
    return target




def warpingUsingBarycentricCoordinate(image1, image2, boundary1=[],boundary2=[],name='output'):

    # transformedTriangle=np.zeros(shape=image2.shape,dtype=np.uint8)
    transformedTriangle=image2.copy()
    min=np.min(boundary1, axis=0)
    max=np.max(boundary1, axis=0)
    minx=int(np.round(min[0]))
    miny=int(np.round(min[1]))
    maxx=int(np.round(max[0]))
    maxy=int(np.round(max[1]))

    for i in range(minx,maxx,1):
        for j in range(miny, maxy,1):
            if(pointInBoundary(i,j,boundary1)):
                W=getBarycentricCoordinates(boundary1, (i,j))
                x,y=getPointUsingBarycentric(boundary2,W[::-1])
                x=int(round(x))
                y=int(round(y))
                transformedTriangle[x,y]=image1[i,j]
    
    # TransformedImage = Image.fromarray(transformedTriangle)
    # TransformedImage.save(name+".jpg")
    return transformedTriangle




def reverseWarpingUsingBarycentricCoordinate(source, target, sourceBoundary=[], targetBoundary=[], name='output', carry=False):
    #since we are using area method to determine if a point is inside a triangle we are calculating area of the triangle priorly
    A = np.round(areaTriangle(targetBoundary[0][0], targetBoundary[0][1],
                      targetBoundary[1][0], targetBoundary[1][1],
                      targetBoundary[2][0], targetBoundary[2][1]))
    
    #carry here means we will be warping multiple triangles by running this function iterratively so use the target image only 
    if(carry):
        transformedTriangle=target.copy()
    else:
        transformedTriangle=np.zeros(shape=target.shape,dtype=np.uint8)

    #finding the bounding box of the target triangle
    min=np.min(targetBoundary, axis=0)
    max=np.max(targetBoundary, axis=0)
    minx=int(np.round(min[0]))
    miny=int(np.round(min[1]))
    maxx=int(np.round(max[0]))
    maxy=int(np.round(max[1]))

    #running for each pixel in the target triangle and mapping it to the source pixel
    for i in range(minx,maxx,1):
        for j in range(miny, maxy,1):
            if(pointInBoundary(i,j,targetBoundary,A)):
                W=getBarycentricCoordinates(targetBoundary, (i,j))
                x,y=getPointUsingBarycentric(sourceBoundary,W)
                x=int(round(x))
                y=int(round(y))
                # print(transformedTriangle[i,j],source[x,y])
                transformedTriangle[i,j]=source[x,y]
    
    # TransformedImage = Image.fromarray(transformedTriangle)
    # TransformedImage.save(name+".jpg")
    return transformedTriangle

