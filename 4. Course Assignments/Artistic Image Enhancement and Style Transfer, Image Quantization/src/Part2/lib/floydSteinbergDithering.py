import numpy as np
# from first.helper import showImage

def applyDithering(original_image, quantized_image):
    quantization_error= original_image-quantized_image
    shape=quantized_image.shape
    quantization_error1=quantization_error*7/16
    quantization_error2=quantization_error*3/16
    quantization_error3=quantization_error*5/16
    quantization_error4=quantization_error*1/16
    for j in range(0,shape[1]-1):
        for i in range(1,shape[0]-1):
            quantized_image[i+1,j] += quantization_error1[i,j]
            quantized_image[i-1,j+1] += quantization_error2[i,j]
            quantized_image[i,j+1] += quantization_error3[i,j]
            quantized_image[i+1,j+1] += quantization_error4[i,j]
    return np.clip(quantized_image,0,1)
