
import numpy as np
from warping import areaTriangle, pointInBoundary
from PIL import Image

def getMask(mask,boundary):
    A = np.round(areaTriangle(boundary[0][0], boundary[0][1],
                      boundary[1][0], boundary[1][1],
                      boundary[2][0], boundary[2][1]))
    
    min=np.min(boundary, axis=0)
    max=np.max(boundary, axis=0)
    minx=int(np.round(min[0]))
    miny=int(np.round(min[1]))
    maxx=int(np.round(max[0]))
    maxy=int(np.round(max[1]))

    for i in range(minx,maxx,1):
        for j in range(miny, maxy,1):
            if(pointInBoundary(i,j,boundary,A)):
                mask[i,j]=[255,255,255]
    return mask