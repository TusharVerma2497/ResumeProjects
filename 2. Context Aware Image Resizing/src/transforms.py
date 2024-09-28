import numpy as np
import cv2

class DCT:
    
    def transform(self, img_gray):

        M,N=img_gray.shape

        #storing the transformed coeff
        transformed_img=np.zeros(img_gray.shape)

        self.CosineTransformMatrixRow=np.zeros((N,N))
        self.rowConst=np.ones((M,N))
        self.rowConst*=(1/np.sqrt(N))
        self.rowConst[0,:]*=1/(2**0.5)
        for k in range(N):
            for n in range(N):
                self.CosineTransformMatrixRow[k, n]= np.cos(np.pi*( n + 0.5 )* k/N)
        transformed_img= np.matmul(img_gray,self.CosineTransformMatrixRow.T)*self.rowConst

        return transformed_img



class DST:
    
    def __init__(self):
        pass
    
    def transform(self, img_gray, axis=1):

        M,N=img_gray.shape

        #storing the transformed coeff
        transformed_img=np.zeros(img_gray.shape)
        self.sineTransformMatrixCol=np.zeros((M,M))
        self.colConst=np.ones((M,N))
        self.colConst*=(1/np.sqrt(M))
        self.colConst[:,0]*=1/(2**0.5)
        for k in range(M):
            for n in range(M):
                self.sineTransformMatrixCol[k, n]= (np.pi*n/M)*np.sin(np.pi*( k + 0.5 )* n/M)
        transformed_img= np.matmul(self.sineTransformMatrixCol, img_gray)*self.colConst
    
        return transformed_img
    
