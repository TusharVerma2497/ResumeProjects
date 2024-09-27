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
    K=[]
    # Number of training examples
    m=len(Y)
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
    print("Time in seconds: {}".format(time.time()-st))

    lambdas=np.ravel(sol['x'])

    indices=np.where((lambdas > 1e-5) & (lambdas <=C))[0] #index of support vectors
    lambdas = lambdas[indices] # sv lamdas
    supportV = X[indices] # sv X
    supportVLabel = Y[indices] # sv Y

    # Displaying top 5 SV
    displayTopSV(lambdas)
    print("\nImages of Top-5 coefficients saved")

    with open('supportVectorsgaussian', 'wb') as f:
        pickle.dump(indices,f)

    #finding b value
    b=0
    for i in range(len(lambdas)):
        t=np.zeros(len(lambdas))
        for j in range(len(lambdas)):
            t[j]=K[indices[i],indices[j]]
        # t.reshape(len(lambdas))
        b+=supportVLabel[i]-np.sum(lambdas*supportVLabel*t)
    b/=len(lambdas)
    print("Value of Intercept: "+str(b))
    print("Number of Support Vectors: "+str(len(supportV)))
    return {'lambdas':lambdas,'SV':supportV,'SVLabels':supportVLabel,'intercept':b}

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
# with open('../part2_data/train_data.pickle', 'rb') as file:
with open(os.path.join(trainDataDir,'train_data.pickle'), 'rb') as file:
    training=pickle.load(file)

# my entry number is 2022aiy7514
# last digit is 4, so the classes I have to train my model is 4 and 0
# taking subset of these 2 classes only
# Also changing the class labels from (0,4) to (-1,+1) 

data={'data':[],'labels':[]}
for i in range(len(training["data"])):
    if(training["labels"][i]==0 or training["labels"][i]==4):
        # data['data'].append(np.asarray(training["data"][i].reshape(3072,order='C'),dtype='int32'))
        data['data'].append(np.asarray(list(map(lambda x:x/255.0,training["data"][i].reshape(3072,order='C'))),dtype='float64'))
        if(training["labels"][i]==0):
            data['labels'].append(-1.0)
        else:
            data['labels'].append(1.0)


# defining linear kernel
def gaussianKernel(a,b,gamma=0.001):
    t=a-b
    t=np.linalg.norm(t)**2
    return np.exp(-gamma*t)


model=SVM(X=np.array(data['data']),Y=np.array(data['labels']),kernel=gaussianKernel,C=1.0)


# saving the solution to the convex optimization to the file
with open('cvxoptest','wb') as file:
    pickle.dump(model,file)

# Reading solution from the pickle file
with open('cvxoptest','rb') as file:
    model=pickle.load(file)   

def predict(pixels):
    s=0
    for j in range(len(model['SV'])):
        s+=model['lambdas'][j]*model['SVLabels'][j]*gaussianKernel(pixels,model['SV'][j])
    return s+model['intercept']


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
        data['data'].append(np.asarray(list(map(lambda x:x/255.0,training["data"][i].reshape(3072,order='C'))),dtype='float64'))
        if(training["labels"][i]==0):
            data['labels'].append(-1.0)
        else:
            data['labels'].append(1.0)

confution=[[0,0],[0,0]]
for i in range(len(data['data'])):
    t=predict(data['data'][i])

    if(data['labels'][i]==1):
        # True Positive
        if(t>0):
            confution[0][0]+=1
        # False Negative
        else:
            confution[1][0]+=1
    else:
        # True Negative
        if(t<=0):
            confution[1][1]+=1
        # False Positive
        else:
            confution[0][1]+=1    
print("Confusion Matrix:")        
print(confution)
print("Accuracy of test data: {}\n".format((confution[0][0]+confution[1][1])*100/len(data['data'])))



# Time in seconds: 64.00805234909058

# Images of Top-5 coefficients saved
# Value of Intercept: -2.1813674081583305
# Number of Support Vectors: 1754
# [[858, 143], [142, 857]]
# Accuracy of test data: 85.75
