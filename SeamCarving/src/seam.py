import numpy as np
import matplotlib.pyplot as plt
from helper import saveImage
import os
import datetime

def dynamicCost(ImportanceMap,h,w):
    costArray=np.zeros((h,w))
    costArray[-1,:]=np.array(ImportanceMap[-1])
    paths=[[] for i in range(w)]
    
    for i in range(h-2,-2,-1):
        # for first column
        m=0 if(costArray[i+1,0]<costArray[i+1,1]) else 1
        if(i<0):
            paths[0]=paths[m][:h-2-i].copy()
            paths[0].append((i+1,m))
        else:
            costArray[i,0] = ImportanceMap[i][0] + costArray[i+1,m]
            paths[0]=paths[m][:h-2-i].copy()
            paths[0].append((i+1,m))
        
        # for last column
        m=-1 if(costArray[i+1,-1]<costArray[i+1,-2]) else -2
        if(i<0):
            paths[-1].append((i+1,m))
        else:
            costArray[i,-1] = ImportanceMap[i][-1] + costArray[i+1,m]
            paths[-1].append((i+1,m))

        #for rest of the columns
        for j in range(1,w-1,1):
            m=j-1 if(costArray[i+1,j-1]<costArray[i+1,j] and costArray[i+1,j-1]<costArray[i+1,j+1]) else j if(costArray[i+1,j]<costArray[i+1,j+1]) else j+1
            if(i<0):
                paths[j]=paths[m][:h-2-i].copy()
                paths[j].append((i+1,m))
            else:
                costArray[i,j] = ImportanceMap[i][j] + costArray[i+1,m]
                paths[j]=paths[m][:h-2-i].copy()
                paths[j].append((i+1,m))
    
    return (costArray,paths)




def getN_DisjointSeams(firstRow,alpha=0.1):
    indices = np.argsort(firstRow)
    standardDeviance=np.std(firstRow)
    limit=alpha*standardDeviance
    for i  in range(len(indices)):
        if(firstRow[indices[i]]>firstRow[indices[0]]+limit):
            limit=i
            break
    minSeams=indices[:limit]
    return minSeams




def getSeamMask(minSeams, paths, removeCount, h, w):
    mask=np.zeros((h,w))
    for i in minSeams:
        if(removeCount==0):
                break
        temp=paths[i]
        clearSeam=True
        for j in temp:
            if(mask[j[0],j[1]]==1):
                clearSeam=False
                break
        if(clearSeam):
            removeCount-=1
            for j in temp:
                mask[j[0],j[1]]=1

    return (mask, removeCount)


def saveProgress(mask,img_RGB, removeCount,axis):
    temp=np.expand_dims(mask, axis=-1)
    temp=1-np.repeat(temp, 3, axis=-1).astype(np.uint8)
    temp=temp*img_RGB
    if(axis==1):
        temp=np.transpose(temp, (1, 0, 2))
    # plt.imshow(temp)
    # plt.show()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saveImage(temp,os.path.join("Part1_results",f"_ts={timestamp}"))


def removeSeam(ImportanceMap, img_RGB, removeCount, alpha=0.1, axis=0, save=False):
    ImportanceMap=ImportanceMap/np.max(ImportanceMap)
    if(axis==1):
        ImportanceMap=ImportanceMap.T
        img_RGB = np.transpose(img_RGB, (1, 0, 2))

    ImportanceMap=ImportanceMap.tolist()
    img_RGB=img_RGB.tolist()

    while(removeCount!=0):
        h,w=len(ImportanceMap),len(ImportanceMap[0])
        # print(f"axis={axis} decreased to: {h},  Seams Left to be removed:{removeCount}")
        
        #dynamic algo for cost calculation based on Importance map
        costArray,paths=dynamicCost(ImportanceMap,h,w)
        # plt.imshow(costArray)
        # plt.show()
        # for i in range(len(paths)):
        #     print(i)
        #     img=np.zeros((h,w))
        #     for j in paths[i]:
        #         # print(j)
        #         img[j[0],j[1]]=255
        #     saveImage(img,os.path.join("Part1_results",f"_w={i}"))
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # getting n minimum disjoint seams maximum 100*alpha% deviation from the min cost seam
        minSeams=getN_DisjointSeams(costArray[0,:],alpha=alpha)
        
        
        mask,removeCount=getSeamMask(minSeams,paths,removeCount, h,w)

        if(save):
            saveProgress(mask, img_RGB, removeCount, axis)
        #removing the pixels
        for i in range(h-1,-1,-1):
            for j in range(w-1,-1,-1):
                if(mask[i,j]==1):
                    ImportanceMap[i].pop(j)
                    img_RGB[i].pop(j)

    if(axis==0):
        return (np.array(img_RGB),np.array(ImportanceMap))
    else:
        return (np.transpose(img_RGB, (1, 0, 2)),np.array(ImportanceMap).T)






# def insertSeam(ImportanceMap, img_RGB, addCount, alpha=0.1, axis=0, save=False):
#     if(axis==1):
#         ImportanceMap=ImportanceMap.T
#         img_RGB = np.transpose(img_RGB, (1, 0, 2))

#     ImportanceMap=ImportanceMap.tolist()
#     img_RGB=img_RGB.tolist()

#     while(addCount!=0):
#         h,w=len(ImportanceMap),len(ImportanceMap[0])
#         # print(f"axis={axis} increased to: {h},  Seams to be added:{addCount}")
        
#         #dynamic algo for cost calculation based on Importance map
#         costArray,paths=dynamicCost(ImportanceMap,h,w)

#         # getting n minimum disjoint seams maximum 100*alpha% deviation from the min cost seam
#         minSeams=getN_DisjointSeams(costArray[0,:],alpha=alpha)
        
#         mask,addCount=getSeamMask(minSeams,paths,addCount, h,w)
#         if(save):
#             saveProgress(mask, img_RGB,addCount,axis)
#         #adding the pixels
#         for i in range(h-1,-1,-1):
#             for j in range(w-1,-1,-1):
#                 if(mask[i,j]==1):
#                     if(j+1<w-1):
#                         newPixelImportanceValue= (ImportanceMap[i][j]+ImportanceMap[i][j+1])//2
#                         newPixelValue=((np.array(img_RGB[i][j])+np.array(img_RGB[i][j+1]))//2).tolist()
#                     if(j-1>0):
#                         newPixelImportanceValue= (ImportanceMap[i][j]+ImportanceMap[i][j-1])//2
#                         newPixelValue=((np.array(img_RGB[i][j])+np.array(img_RGB[i][j-1]))//2).tolist()

                            
#                     ImportanceMap[i].insert(j, newPixelImportanceValue)
#                     img_RGB[i].insert(j, newPixelValue)

#     if(axis==0):
#         return (np.array(img_RGB),np.array(ImportanceMap))
#     else:
#         return (np.transpose(img_RGB, (1, 0, 2)),np.array(ImportanceMap).T)



def insertSeam(ImportanceMap, img_RGB, addCount, alpha=0.1, axis=0, save=False):
    if(axis==1):
        ImportanceMap=ImportanceMap.T
        img_RGB = np.transpose(img_RGB, (1, 0, 2))

    ImportanceMap=ImportanceMap.tolist()
    img_RGB=img_RGB.tolist()

    while(addCount!=0):
        h,w=len(ImportanceMap),len(ImportanceMap[0])
        # print(f"axis={axis} increased to: {h},  Seams to be added:{addCount}")
        
        #dynamic algo for cost calculation based on Importance map
        costArray,paths=dynamicCost(ImportanceMap,h,w)

        # getting n minimum disjoint seams maximum 100*alpha% deviation from the min cost seam
        minSeams=getN_DisjointSeams(costArray[0,:],alpha=alpha)
        
        mask,addCount=getSeamMask(minSeams,paths,addCount, h,w)
        if(save):
            saveProgress(mask, img_RGB,addCount,axis)
        #adding the pixels
        for i in range(h-1,-1,-1):
            for j in range(w-1,-1,-1):
                if(mask[i,j]==1):
                    left=(i,j-1)
                    right=(i,j+1)
                    top=(i-1,j)
                    bottom=(i+1,j)

                    if(j+1>=w-1):
                        right=left
                    if(j-1<=0):
                        left=right
                    
                    if(i+1>=h-1):
                        bottom=top
                    if(i-1<=0):
                        top=bottom

                    newPixelImportanceValue= (ImportanceMap[i][j]+
                                              ImportanceMap[left[0]][left[1]]+
                                              ImportanceMap[right[0]][right[1]]+
                                              ImportanceMap[top[0]][top[1]]+
                                              ImportanceMap[bottom[0]][bottom[1]])//5
                    
                    newPixelValue= (np.array(img_RGB[i][j])+
                                              np.array(img_RGB[left[0]][left[1]])+
                                              np.array(img_RGB[right[0]][right[1]])+
                                              np.array(img_RGB[top[0]][top[1]])+
                                              np.array(img_RGB[bottom[0]][bottom[1]]))//5

                            
                    ImportanceMap[i].insert(j, newPixelImportanceValue)
                    img_RGB[i].insert(j, newPixelValue)

    if(axis==0):
        return (np.array(img_RGB),np.array(ImportanceMap))
    else:
        return (np.transpose(img_RGB, (1, 0, 2)),np.array(ImportanceMap).T)