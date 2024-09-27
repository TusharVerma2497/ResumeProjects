import numpy as np
from helper import performTransformation, getBarycentricCoordinates, pointInBoundary, getPointUsingBarycentric
from PIL import Image



def reverseWarping(warpMatrix, image, boundary=[],name='output'):
    inverse_warpMatrix = np.linalg.inv(np.vstack((warpMatrix,np.array([0,0,1]))))

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
    if len(boundary)==0:
        
        listOfPoints=[]
        listOfColors=[]
        for i in range(minx,maxx,1):
            for j in range(miny, maxy,1):
                listOfPoints.append([i,j])
        listOfPoints=np.float32(listOfPoints)
        invertedPoints=performTransformation(listOfPoints,inverse_warpMatrix)
        for i in invertedPoints:
            x=int(i[0])
            y=int(i[1])
            if(x<0 or y<0 or x>=image.shape[0] or y>=image.shape[1]):
                listOfColors.append(np.uint8([0,0,0]))
            else:
                listOfColors.append(image[x,y])
        widthOfTransformedImage=maxx-minx
        heightOfTransformedImage=maxy-miny
        listOfColors=np.array(listOfColors)
        listOfColors=listOfColors.reshape((widthOfTransformedImage,heightOfTransformedImage,3))
        TransformedImage = Image.fromarray(listOfColors)
        TransformedImage.save(name+".jpg")
    
    
    elif len(boundary)==3:
        minBoundary=(minx, miny)
        finalImage=np.zeros(shape=(maxx-minx, maxy-min,3), dtype=np.uint8)
        # print(finalImage.shape)
        transformedPoints=performTransformation(boundary,warpMatrix)
        #finding the min and max of the the transformed axis
        min=np.min(transformedPoints, axis=0)
        max=np.max(transformedPoints, axis=0)
        minx=int(np.round(min[0]))
        miny=int(np.round(min[1]))
        maxx=int(np.round(max[0]))
        maxy=int(np.round(max[1]))
        # print(transformedPoints)

        listOfPoints=[]
        listOfColors=[]
        # finalImage=image.copy()
        
        for i in range(minx,maxx,1):
            for j in range(miny, maxy,1):
                # print(i,j)
                if(pointInBoundary(i,j,transformedPoints)):
                    listOfPoints.append([i,j])

        
        listOfPoints=np.float32(listOfPoints)
        invertedPoints=performTransformation(listOfPoints,inverse_warpMatrix)
        # invertedPoints=performTransformation(listOfPoints,warpMatrix)

        for i in invertedPoints:
            x=int(i[0])
            y=int(i[1])
            if(x<0 or y<0 or x>=image.shape[0] or y>=image.shape[1]):
                listOfColors.append(np.uint8([0,0,0]))
            else:
                listOfColors.append(image[x,y])


        for i in range(len(listOfPoints)):
            x=int(listOfPoints[i,0])
            y=int(listOfPoints[i,1])
            # print(x,y)
            finalImage[x-minBoundary[0],y-minBoundary[1]]=listOfColors[i]
            # finalImage[x,y]=np.uint8([0,0,0])
        TransformedImage = Image.fromarray(finalImage)
        TransformedImage.save(name+".jpg")
    return TransformedImage


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
    
    TransformedImage = Image.fromarray(transformedTriangle)
    TransformedImage.save(name+".jpg")
    return transformedTriangle




def reverseWarpingUsingBarycentricCoordinate(image1, image2, boundary1=[],boundary2=[],name='output'):

    # transformedTriangle=np.zeros(shape=image2.shape,dtype=np.uint8)
    transformedTriangle=image2.copy()
    min=np.min(boundary2, axis=0)
    max=np.max(boundary2, axis=0)
    minx=int(np.round(min[0]))
    miny=int(np.round(min[1]))
    maxx=int(np.round(max[0]))
    maxy=int(np.round(max[1]))

    for i in range(minx,maxx,1):
        for j in range(miny, maxy,1):
            if(pointInBoundary(i,j,boundary2)):
                W=getBarycentricCoordinates(boundary2, (i,j))
                x,y=getPointUsingBarycentric(boundary1,W)
                x=int(round(x))
                y=int(round(y))
                transformedTriangle[i,j]=image1[x,y]
    
    TransformedImage = Image.fromarray(transformedTriangle)
    TransformedImage.save(name+".jpg")
    return transformedTriangle
