import numpy as np


def rgb_to_hsi(image):

    #defining the const matrix
    matrix=np.array([[1/3, 1/3, 1/3],
                     [-np.sqrt(6)/6, -np.sqrt(6)/6, np.sqrt(6)/3],
                     [1/np.sqrt(6), -2/np.sqrt(6), 0.]])

    image=np.dot(image, matrix)
    I,V1,V2=np.dsplit(image,image.shape[-1])
    S=np.sqrt(np.power(V1,2), np.power(V2,2))
    H = np.where(V1 != 0, np.arctan(V2 / V1), 0.)
    return H,S,I


def r_map(h,i):
    return (h+1)/(i+1)


def computeThreshold(rmap):
    t , c = np.unique(rmap, return_counts=True)
    counts=np.zeros(256, dtype=int)
    j=0
    for i in range(256):
        if(j==len(t)):
            break
        if(i<t[j]):
            pass
        else:
            counts[i]=c[j]
            j+=1
    prob_matrix=counts/np.sum(counts)
    minT=0
    minTValue=+1234567890
    W1=0
    W2=sum(prob_matrix)
    numbers256=np.array([i for i in range(256)])
    wSum1=0
    wSum2=np.sum(prob_matrix*numbers256)

    for i in range(0,256,1):
        W1=W1+prob_matrix[i]
        W2=W2-prob_matrix[i]
        
        temp=i*prob_matrix[i]
        wSum1=wSum1+temp
        wSum2=wSum2-temp

        #calculating mu1 and mu2
        mu1=wSum1/W1
        mu2=wSum2/W2
        
        #computing equation 6
        term1=np.sum(prob_matrix[0:i+1]*np.power((numbers256[0:i+1]-mu1),2))
        term2=np.sum(prob_matrix[i+1:]*np.power((numbers256[i+1:]-mu2),2))
        eq6=term1+term2
        if(minTValue> eq6):
            minTValue=eq6
            minT=i
    return minT



def computeShadowMap(rmap, T):
    return np.where(rmap > T, 1, 0)


def merge(sm,image,l):
    # making shadowed image same size as input image by cloning in the channel dimension
    sm=np.repeat(sm,3,2)
    return np.where(sm == 0, l*image + (1-l)*sm, image)