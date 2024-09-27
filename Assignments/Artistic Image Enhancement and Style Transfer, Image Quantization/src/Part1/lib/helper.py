import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import os

def normalizeBy255(image):
    return image/255


def changeRange(image, newMin, newMax):
    currentMax=np.nanmax(image)
    currentMin=np.nanmin(image)
    return np.interp(image, (currentMin, currentMax), (newMin, newMax))


def showImage(image, name='img', cmap="viridis",):
    plt.axis('off')
    plt.title(name)
    plt.imshow(image, cmap)
    plt.show()



def plotMultiImage(lis, titles, figsize=(10, 20),l=6,b=2):
    fig, axes = plt.subplots(l, b, figsize=figsize)
    for i in range(l):
        for j in range(b):
            axes[i, j].set_title(str(titles[i*b+j]))
            axes[i, j].imshow(lis[i*b+j], cmap='gray')  # Plot the image
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def readImage(path):
    image = Image.open(path)
    image = np.asarray(image)
    return image

def saveImage(image, name="", path="generated_images", part=1):
    path='Part'+str(part)+'_'+path
    if not os.path.exists(path):
        os.makedirs(path)
    pil_image = Image.fromarray(image)
    pil_image.save(os.path.join(path, name+str(datetime.now())+".jpg"))

def BGRtoLAB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

def RGBtoLAB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

def LABtoRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

def RGBtoGRAY(image):
    if image.shape[-1]==1:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def readImageCV2(path):
    return cv2.imread(path)

def GRAYtoRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def GRAYtoLAB(image):
    if image.shape[-1]==3:
        return RGBtoLAB(image)[:,:,0]
    elif image.shape[-1]==1:
        image=GRAYtoRGB(image)
        return RGBtoLAB(image)[:,:,0]
    

def calculate_mode(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    modes = unique_values[counts == max_count]
    return modes[0]


# implementing binary search for finding the nearest sample
def binary_search(target,samples):
    target=float(target)
    left, right = 0, len(samples) - 1
    leftClosest=left
    rightClosest=right
    
    while left < right:
        mid = left + (right - left) // 2

        if samples[mid][0] < target:
            left = mid + 1
            leftClosest=left
        else:
            right = mid - 1
            rightClosest=right

    #if perfect match not found the returning the closest one of the leftClosest and rightClosest
    return leftClosest if target-samples[leftClosest][0] <= samples[rightClosest][0]-target else rightClosest


def showHistogramGray(image):
    hist=cv2.calcHist([image],[0],None,[256],[0,255])
    plt.plot(hist)
    plt.show()