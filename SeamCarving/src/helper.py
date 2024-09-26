from PIL import Image
import numpy as np
import cv2
from datetime import datetime

def readImage(path):
    image = Image.open(path)
    image = np.asarray(image)
    return image

def saveImage(image, path):
    if(len(image.shape)==2):
         pil_image = Image.fromarray(image)
         pil_image = pil_image.convert('L') 
    else:
         image=image.astype(np.uint8)
         pil_image= Image.fromarray(image)
    pil_image.save(path+".jpg")

def RGBtoYCbCr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


def RGBtoLAB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)


def RGBtoGRAY(image):
    if image.shape[-1]==1:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def makeShape8Divisible(img_gray):
    #first padding it make w and h perfectily divisible by 8
        w,h=img_gray.shape
        reqPadw=(8-w%8)%8
        reqPadh=(8-h%8)%8

        # required padding in all the 4 directions
        padTop=reqPadw//2
        padBottom=reqPadw-padTop
        padLeft=reqPadh//2
        padRight=reqPadh-padLeft

        #gray image after padding
        img_gray=np.pad(img_gray, ((padTop, padBottom), (padLeft, padRight)), mode='reflect')
        # print(img_gray.shape)

        return img_gray


def changeRange(image, newMin, newMax):
    currentMax=np.nanmax(image)
    currentMin=np.nanmin(image)
    return np.interp(image, (currentMin, currentMax), (newMin, newMax))



def removePadding(img_gray, w, h):
        
        paddedW, paddedH=img_gray.shape
        reqPadw=(8-w%8)%8
        reqPadh=(8-h%8)%8
        # required padding in all the 4 directions
        padTop=reqPadw//2
        padBottom=reqPadw-padTop
        padLeft=reqPadh//2
        padRight=reqPadh-padLeft

        return img_gray[padTop:paddedW-padBottom, padLeft:paddedH-padRight]


def sharpenImage(image):
    sharpened = cv2.Laplacian(image, cv2.CV_64F)
    # Convert the result to uint8 (8-bit) image
    sharpened = np.uint8(np.abs(sharpened))
    return sharpened