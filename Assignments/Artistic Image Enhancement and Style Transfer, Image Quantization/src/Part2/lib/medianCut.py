import numpy as np
from .dataStructures import bucket
import math
import heapq

def quantize(image, numOfColors):
    finalBuckets=[]

    #creating a numpy array of pixels and their positional tuple
    total_pixels=image.shape[0]*image.shape[1]
    pixels=image.reshape((total_pixels,image.shape[2]))
    row_indices, col_indices = np.indices(image.shape[:2])
    position_tuples = np.stack((row_indices, col_indices), axis=-1)
    position_tuples=position_tuples.reshape((total_pixels,2))

    # creating initial bucket
    b0=bucket(pixels=pixels,pos=position_tuples)


    # Apply divide and conqure
    # if number of colors is in power of 2
    if(numOfColors > 0 and (numOfColors & (numOfColors - 1)) == 0):
        depth=int(math.log(numOfColors,2))

        #defining a nested D&C recursive function
        def divideAndConqure(mainBucket:bucket,i=0):
            if(i==depth):
                finalBuckets.append(mainBucket)
            else:
                leftBucket, rightBucket= mainBucket.splitMedianOn_mostDynamicChannel()
                divideAndConqure(leftBucket, i+1)
                divideAndConqure(rightBucket, i+1)

        #calling the nested function 
        divideAndConqure(b0)

    #if number of colors is not in power of 2
    else:
        # pushing the b0 bucket
        heapq.heappush(finalBuckets,b0)
        while(len(finalBuckets)<=numOfColors):
            largest = heapq.heappop(finalBuckets)
            leftBucket, rightBucket= largest.splitMedianOn_mostDynamicChannel()
            heapq.heappush(finalBuckets,leftBucket)
            heapq.heappush(finalBuckets,rightBucket)


    quantizedImage=np.empty_like(image)    
    for i in finalBuckets:
        for j in range(len(i)):
            quantizedImage[i.pos_[j][0],i.pos_[j][1]]=i.avgPixel
    
    return quantizedImage

