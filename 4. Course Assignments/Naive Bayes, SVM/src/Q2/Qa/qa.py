import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from cvxopt import solvers, matrix
import time
import sys
import os


# location of training and test data
trainDataDir=sys.argv[1]
testDataDir=sys.argv[2]


# This is the right one
def SVM(X,Y,kernel,C):
    # Number of training examples
    m=len(Y)
    K=[]
    for i in range(m):
        for j in range(m):
            K.append(kernel(X[i],X[j]))
    K=np.array(K).reshape(m,m)
    P=matrix(np.outer(Y,Y)*K)
    q=np.ones(m)
    q=matrix(q*-1)
    t1=np.identity(m)
    t2=t1*-1
    G=matrix(np.append(t2,t1).reshape(2*m,m))
    h=matrix(np.append(np.zeros(m),np.ones(m)*C))
    b=matrix(np.array([0.0]))
    A=np.array(Y).reshape(1,m)
    A=matrix(A,tc='d')

    print('Parameters calculated')
    st=time.time()
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h,A,b)
    print("Time taken in seconds: {}".format(time.time()-st))

    lambdas=np.ravel(sol['x'])
    temp=np.where((lambdas > 1e-5) & (lambdas <=C))[0]

    # Displaying top 5 SV
    displayTopSV(lambdas[temp])
    print("Images of Top-5 coefficients saved")

    # finding w
    w=0
    for i in temp:
        w+=(lambdas[i]*Y[i])*X[i]
    print("\nw: {}".format(w))

    # finding b (lambdas whose value is between 0 and C are the support vectors)
    temp2=np.where((lambdas>1e-5) &(lambdas<C))[0]
    b=0
    for i in temp2:
        b+=Y[i]-X[i].dot(w)
    b/=len(temp2)

    with open('supportVectorsLinear', 'wb') as f:
        pickle.dump(temp2,f)
    
    print("b: "+str(b))
    print("Number of support vectors: "+ str(len(temp2)))
    return (w,b)

def displayTopSV(SV):
    sortedSV=sorted(SV, reverse=True)
    #Top 5 fetures
    topFive=sortedSV[0:6]
    for i in range(len(topFive)):
        a=np.where(topFive[i]==SV)[0][0]
        img=np.array(list(map(lambda x:min(int(x*255),255),data['data'][a])))
        img=img.reshape(32,32,3)
        plt.axis("off")
        plt.imshow(img)
        plt.savefig("img"+str(5-i))

# Reading training data from pickle
with open(os.path.join(trainDataDir,'train_data.pickle'), 'rb') as file:
    training=pickle.load(file)

# my entry number is 2022aiy7514
# last digit is 4, so the classes I have to train my model is 4 and 0
# taking subset of these 2 classes only
# Also changing the class labels from (0,4) to (-1,+1) 

data={'data':[],'labels':[]}
for i in range(len(training["data"])):
    if(training["labels"][i]==0 or training["labels"][i]==4):
        # print(training["labels"][i])
        # data['data'].append(np.asarray(training["data"][i].reshape(3072,order='C'),dtype='int32'))
        data['data'].append(np.asarray(list(map(lambda x:x/255, training["data"][i].reshape(3072,order='C'))),dtype='float64'))
        if(training["labels"][i]==0):
            data['labels'].append(-1.0)
        else:
            data['labels'].append(1.0)

# defining linear kernel
def linearKernel(a,b):
    return a.dot(b)

res=SVM(X=np.array(data['data']),Y=np.array(data['labels']),kernel=linearKernel,C=1.0)
w,b=res

with open('wbMymodel','wb') as f:
    pickle.dump((b,w),f)

# saving the solution to the convex optimization to the file
with open('cvxoptest','wb') as file:
    pickle.dump(res,file)

# Reading solution from the pickle file
with open('cvxoptest','rb') as file:
    w,b=pickle.load(file)

# Reshaping and displaying w
minW=min(w)
maxW=max(w)
pixelsW=np.array(list(map(lambda x: int(((x-minW)*(255))/(maxW-minW)),w)))
pixelsW=pixelsW.reshape(32,32,3)
plt.axis("off")
plt.imshow(pixelsW)
plt.savefig('w')


def predict(pixels):
    val=pixels.dot(w)+b
    return val


#Testing data
with open(os.path.join(testDataDir,'test_data.pickle'), 'rb') as file:
    training=pickle.load(file)

# my entry number is 2022aiy7514
# last digit is 4, so the classes I have to train my model is 4 and 0
# taking subset of these 2 classes only
# Also changing the class labels from (0,4) to (-1,+1) 

data={'data':[],'labels':[]}
for i in range(len(training["data"])):
    if(training["labels"][i]==0 or training["labels"][i]==4):
        # print(training["labels"][i])
        data['data'].append(np.asarray(list(map(lambda x:x/255, training["data"][i].reshape(3072,order='C'))),dtype='float64'))
        if(training["labels"][i]==0):
            data['labels'].append(-1.0)
        else:
            data['labels'].append(1.0)

def imagesave(pixels,st):
    img=np.array(list(map(lambda x:min(int(x*255),255),pixels)))
    img=img.reshape(32,32,3)
    plt.axis("off")
    plt.imshow(img)
    plt.savefig(st)


confution=[[0,0],[0,0]]
for i in range(len(data['data'])):
    t=predict(data['data'][i])
    # if((t>0 and data['labels'][i]==1) or (t<=0 and data['labels'][i]==-1)):
    #     correct+=1

    if(data['labels'][i]==1):
        # True Positive
        if(t>0):
            # imagesave(data['data'][i],"correctImg/TP"+str(i))
            confution[0][0]+=1
        # False Negative
        else:
            # imagesave(data['data'][i],"incorrectImg/FN"+str(i))
            confution[1][0]+=1
    else:
        # True Negative
        if(t<=0):
            # imagesave(data['data'][i],"correctImg/TN"+str(i))
            confution[1][1]+=1
        # False Positive
        else:
            # imagesave(data['data'][i],"incorrectImg/FP"+str(i))
            confution[0][1]+=1    
print("Confusion Matrix:")        
print(confution)
print("Accuracy of test data: {}\n".format((confution[0][0]+confution[1][1])*100/len(data['data'])))



# Time taken in seconds: 62.71004605293274
# Images of Top-5 coefficients saved

# w: [-0.40311321 -0.09715665 -0.99772818 ... -0.48383869  0.04012286
#  -0.53011407]
# b: 1.5796823291522466
# Number of support vectors: 1505
# [[792, 212], [208, 788]]
# Accuracy of test data: 79.0
