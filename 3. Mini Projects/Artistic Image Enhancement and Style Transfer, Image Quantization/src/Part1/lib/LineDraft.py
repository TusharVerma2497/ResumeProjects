import numpy as np
import cv2
from scipy.signal import convolve2d

sobelY=[[-1,0,1],[-2,0,2],[-1,0,1]]
sobelX=[[-1,-2,-1],[0,0,0],[1,2,1]]

def toGrayScale(image):
   if(len(image.shape)==2):
       return image
   return np.mean(image, axis=-1)

def bilateral_filtering_fast(image, diameter, sigma_color, sigma_space):
    cv2_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return cv2_image

def bilateral_filtering_slow(image, diameter, sigma_color, sigma_space):
    print(image.shape)
    height, width = image.shape
    filtered_image = np.zeros_like(image)
    
    for y in range(height):
        print(y)
        for x in range(width):
            pixel_intensity = image[y, x]
            weighted_sum = 0
            total_weight = 0
            for j in range(y - diameter, y + diameter + 1):
                for i in range(x - diameter, x + diameter + 1):
                    if 0 <= j < height and 0 <= i < width:
                        neighbor_intensity = image[j, i]
                        spatial_distance = np.sqrt((x - i)**2 + (y - j)**2)
                        intensity_difference = np.abs(pixel_intensity - neighbor_intensity)
                        
                        weight = np.exp(-(spatial_distance**2) / (2 * sigma_space**2))
                        weight *= np.exp(-(intensity_difference**2) / (2 * sigma_color**2))
                        
                        weighted_sum += neighbor_intensity * weight
                        total_weight += weight
            
            filtered_image[y, x] = weighted_sum / total_weight
    
    return filtered_image



def applySobel(image):
    Ix = convolve2d(image, sobelX, mode='same')
    Iy = convolve2d(image, sobelY, mode='same')
    return (Ix,Iy)



def computelineDraft(image, T):
    return np.where(image > T, 1, 0)