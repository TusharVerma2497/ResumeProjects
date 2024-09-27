import cv2
import matplotlib.pyplot as plt
from lib.my_bilinear_interpolation import resize
from lib.compare import rms_distance ,L1_distance, getChangeMask
import numpy as np
from PIL import Image
import sys
import os
from datetime import datetime
# from chatGPT import bilinear_interpolation



def executeAssignment(new_size, img_loc, metric='RMS', library='opencv', changeThreshold=[None, 15, 30]):
    # new_size = (400,400)

    # Perform bilinear interpolation using opencv or pillow
    if(library=='opencv'): 
        image = cv2.imread(img_loc)
        interpolated_image = cv2.resize(image, new_size[::-1], interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        interpolated_image = cv2.cvtColor(interpolated_image, cv2.COLOR_BGR2RGB)

    elif(library=='pil'):
        image = Image.open(img_loc)
        interpolated_image = image.resize(new_size[::-1], Image.BILINEAR)
        image = np.asarray(image)
        interpolated_image = np.asarray(interpolated_image)

    channels=image.shape[2]

    # Perform bilinear interpolation using my implementation of the same
    my_interpolated_image=np.zeros(shape=(new_size[0],new_size[1],channels))
    for i in range(channels):
        my_interpolated_image[:,:,i]=resize(image[:,:,i],new_size)
        my_interpolated_image=np.round(my_interpolated_image).astype(np.int32)

    # my_interpolated_image=bilinear_interpolation(image,new_size[0],new_size[1])

    print(f'rms difference between the images = {rms_distance(interpolated_image, my_interpolated_image)}')
    print(f'L1 distance between the images = {L1_distance(interpolated_image, my_interpolated_image)}')
    changeMask1=getChangeMask(interpolated_image, my_interpolated_image, threshold=changeThreshold[0], metric=metric)
    changeMask2=getChangeMask(interpolated_image, my_interpolated_image, threshold=changeThreshold[1], metric=metric)
    changeMask3=getChangeMask(interpolated_image, my_interpolated_image, threshold=changeThreshold[2], metric=metric)
    # print(changeMask1)
    plt.figure(figsize=(12,6))
    plt.subplot(2, 3, 1)
    plt.axis('off')
    plt.title(f'original {image.shape}')
    plt.imshow(image)
    plt.subplot(2, 3, 2)
    plt.axis('off')
    plt.title(f'inbuilt interpolation {(new_size[0],new_size[1],channels)}')
    plt.imshow(interpolated_image)
    plt.subplot(2, 3, 3)
    plt.axis('off')
    plt.title(f'my interpolation {(new_size[0],new_size[1],channels)}')
    plt.imshow(my_interpolated_image/255)
    plt.subplot(2, 3, 4)
    plt.axis('off')
    plt.title(f'{metric} change mask (threshold={changeThreshold[0]})')
    plt.imshow(changeMask1, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.axis('off')
    plt.title(f'{metric}  change mask (threshold={changeThreshold[1]})')
    plt.imshow(changeMask2, cmap='gray')
    plt.subplot(2, 3, 6)
    plt.axis('off')
    plt.title(f'{metric}  change mask (threshold={changeThreshold[2]})')
    plt.imshow(changeMask3, cmap='gray')
    plt.savefig(os.path.join('results','output'+str(datetime.now())+'.jpg'))
    plt.show()


if __name__ == "__main__":

    new_size=(int(sys.argv[2]), int(sys.argv[3]))
    print(new_size)
    img_loc=sys.argv[1]

    #threshold for change pixel
    changeThreshold=[None, 15, 30]

    # metic for calculating the difference between the inbuild library and my implementation of bilinear interpolation
    # values can be {RMS, L1}
    metric='RMS'

    # inbuild library to be used
    # values can be {pil, opencv}
    library='opencv'

    executeAssignment(new_size, img_loc, metric, library, changeThreshold)