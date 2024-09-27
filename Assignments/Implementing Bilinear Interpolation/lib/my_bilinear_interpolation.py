import numpy as np



def calculate(leftExtreme, rightExtreme, pos):
    temp=leftExtreme*(1-pos) + rightExtreme*pos
    if(temp<0):
        temp=0
    if(temp>255):
        temp=255
    return temp



def columnInterpolate(image, shape_new_image, shape_original_image):

    spacingb=(shape_original_image[1]-1)/(shape_new_image[1]-1)

    #expanding rows
    if(shape_new_image[1]>shape_original_image[1]):
        interpolated_rows=list()
        for row in range(shape_original_image[0]):
            i=0
            fraction=spacingb
            l=list()
            l.append(image[row][0])
            while(i+1!=shape_original_image[1] and len(l)<shape_new_image[1]-1):
                l.append(calculate(image[row][i],image[row][i+1],fraction))
                fraction+=spacingb
                if(fraction>1):
                    fraction=fraction%1.0
                    i+=1
            l.append(image[row][-1])
            interpolated_rows.append(l)
        image=np.array(interpolated_rows)
        image=image.reshape((shape_original_image[0],shape_new_image[1]))

    #shrinking rows
    else:
        interpolated_rows=list()
        for row in range(shape_original_image[0]):
            fraction=0
            l=list()
            while(len(l)<shape_new_image[1]):
                rounded_fraction=np.round_(fraction%1.0, decimals = 3)
                if(rounded_fraction==0.0):
                    l.append(image[row][int(fraction)])
                else:
                    i=int(np.floor(fraction))
                    l.append(calculate(image[row][i],image[row][i+1],rounded_fraction))
                fraction+=spacingb
            interpolated_rows.append(l)
        image=np.array(interpolated_rows)
        image=image.reshape((shape_original_image[0],shape_new_image[1]))

    return image



def rowInterpolate(image, shape_new_image, shape_original_image):

    spacingl=(shape_original_image[0]-1)/(shape_new_image[0]-1)
    new_image=np.zeros(shape=shape_new_image)

    #expanding columns
    if(shape_new_image[0]>shape_original_image[0]):
        #copying first and last row
        new_image[0]=image[0]
        new_image[-1]=image[-1]

        #interpolating all the columns
        for column in range(shape_new_image[1]):
            temp=1
            i=0
            fraction=spacingl
            while(i+1!=shape_original_image[0]):
                new_image[temp][column]=calculate(image[i][column],image[i+1][column],fraction)
                temp+=1
                fraction+=spacingl
                if(fraction>1):
                    fraction=fraction%1.0
                    i+=1
    
    #shrinking columns
    else:
        # #interpolating all the columns
        for column in range(shape_new_image[1]):
            fraction=0
            temp=0
            while(temp<shape_new_image[0]):
                rounded_fraction=np.round_(fraction%1.0, decimals = 3)
                if(rounded_fraction==0.0):
                    new_image[temp][column]=image[int(fraction)][column]
                    temp+=1
                else:
                    i=int(np.floor(fraction))
                    new_image[temp][column]=calculate(image[i][column],image[i+1][column],rounded_fraction)
                    temp+=1
                fraction+=spacingl
    return new_image




def resize(image, shape_new_image):

    shape_original_image=image.shape
    image=columnInterpolate(image, shape_new_image, shape_original_image)
    image=rowInterpolate(image, shape_new_image, shape_original_image)

    return image



# original_image=np.array([[3,1,8,5,8],[14,0,20,54,9],[2,1,10,0,11],[6,7,13,4,12], [16,27,3,14,2]])
# print(resize(original_image,(5,4)))