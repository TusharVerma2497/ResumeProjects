import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, markers
from mpl_toolkits import mplot3d
from functools import reduce
from scipy.stats import multivariate_normal
import math
import time
import threading

#reading the commandline arguments
trainDataDir=sys.argv[1]
testDataDir=sys.argv[2]


# reading data
data=pd.read_csv(trainDataDir+"/X.csv",header=None)
X_1=np.array(data.iloc[:,0])
X_2=np.array(data.iloc[:,1])

# normalizing x1 and x2
X_1=((X_1-X_1.mean())/X_1.std()).reshape(len(X_1),1)
X_2=((X_2-X_2.mean())/X_2.std()).reshape(len(X_1),1)
X=np.array([X_1,X_2])
Y=pd.read_csv(trainDataDir+"/Y.csv",header=None)


#       calculaing parameters for category='Alaska'

# indicator function
Y=np.array(list(map(lambda x: 1 if x=='Alaska' else 0,Y.iloc[:,0]))).reshape(len(X_1),1)
# calculating phi
phi1=reduce(lambda x,y: x+y, Y)/len(Y)
# calculating mean mu
mu1=np.matmul(Y.T,X)/sum(Y)

temp=(X-mu1).reshape(2,len(X_1))
t=Y.T*temp

# covariance matrix
coVar=(np.matmul(t,temp.T))/sum(Y)


#       calculaing parameters for category='Canada'

# indicator function
Y1=(Y-1)*-1
# calculating phi
phi2=1-phi1
# calculating mean mu
mu2=np.matmul(Y1.T,X)/sum(Y1)
temp2=(X-mu2).reshape(2,len(X_1))
t=Y1.T*temp2
# covariance matrix
coVar2=(np.matmul(t,temp2.T))/sum(Y1)

def predictClass(X,params):
    l=[]
    for j in range(len(X[0])):
        max=-1
        cat=0
        for i in range(len(params)):
            x=np.array([X[0][j], X[1][j]])
            temp=np.matmul(params[i][2],x)
            temp=np.matmul(x.T,temp)
            Del=-0.5*(temp + math.log(np.linalg.det(params[i][1]))) + math.log(params[i][3])
            if(max<Del):
                Del=max
                cat=i
        l.append(cat)
    return np.array(list(map(lambda x: 'Alaska' if x==1 else 'Canada',l)))
            # temp=(X.reshape(2,100)-params[i][0].reshape(2,1)[0])
            # temp2=np.matmul(np.linalg.inv(params[i][1]), temp)
            # temp=np.matmul(temp.T,temp2)
            # print(temp.shape)
            # pass

# data=pd.read_csv(trainDataDir+"/q4x.dat",header=None,sep='  ',engine='python')
data=pd.read_csv(testDataDir+"/X.csv",header=None)
X_1=np.array(data.iloc[:,0])
X_2=np.array(data.iloc[:,1])

# normalizing x1 and x2
X_1=((X_1-X_1.mean())/X_1.std()).reshape(len(X_1),1)
X_2=((X_2-X_2.mean())/X_2.std()).reshape(len(X_1),1)

y=predictClass(X=np.array([X_1,X_2]),  params=[[mu1,coVar,np.linalg.inv(coVar),phi1],[mu2,coVar2,np.linalg.inv(coVar2),phi2]])
pd.DataFrame(y).to_csv('result_4.txt',index=False,header=None)


#          Drawing the contours and boundary line

# fig=plt.figure(figsize=(9,9))
# for i in range(len(Y)):
#     if(Y[i]==0):
#         plt.scatter(X[0][i],X_2[i], marker='o', color='red')
#     else:
#         plt.scatter(X[0][i],X_2[i], marker='x', color='blue')
# x=np.linspace(min(X_1),max(X_1),50)
# y=np.linspace(min(X_1),max(X_1),50)
# x,y=np.meshgrid(x,y)
# p=np.dstack((x,y))
# r1=multivariate_normal(mu1.reshape(2),coVar)
# r2=multivariate_normal(mu2.reshape(2),coVar)
# z=r1.pdf(p)
# plt.contour(x,y,z,alpha=0.4)
# z=r2.pdf(p)
# plt.contour(x,y,z,alpha=0.4)

# mu1=mu1.reshape(1,2)[0]
# mu2=mu2.reshape(1,2)[0]

# #Plotting the decision boundary
# w0=np.matmul(np.linalg.inv(coVar),mu1)
# w1=np.matmul(np.linalg.inv(coVar),mu2)
# wk0=np.matmul(mu1,w0)
# wk1=np.matmul(mu2,w1)

# temp=w0-w1
# # calulating the slope of the decison boundary
# slope=-temp[0]/temp[1]
# # calulating the slope of the decison boundary
# c=0.5*(wk1-wk0)/(temp[1])

# x=np.array([min(X[0]),max(X[0])])
# y=slope*x+c
# plt.plot(x,y,c='black')
# plt.show()
# # fig.savefig('figCate')








# for i in range(len(Y)):
#     if(Y[i]==0):
#         plt.scatter(X[0][i],X_2[i], marker='o', color='red')
#     else:
#         plt.scatter(X[0][i],X_2[i], marker='x', color='blue')
# x=np.linspace(min(X_1),max(X_1),50)
# y=np.linspace(min(X_1),max(X_1),50)
# x,y=np.meshgrid(x,y)
# p=np.dstack((x,y))
# r1=multivariate_normal(mu1.reshape(2),coVar)
# r2=multivariate_normal(mu2.reshape(2),coVar2)
# z=r1.pdf(p)
# plt.contour(x,y,z,alpha=0.4)
# z=r2.pdf(p)
# plt.contour(x,y,z,alpha=0.4)


# mu1=mu1.reshape(1,2)[0]
# mu2=mu2.reshape(1,2)[0]
# #calculating mu.T*SigmaInverse*mu
# inv_coVar=np.linalg.inv(coVar)
# wk=np.matmul(inv_coVar,mu1.T)
# # print(wk)
# const=np.matmul(mu1,wk)  + math.log(np.linalg.det(coVar))

# for x in np.linspace(-5.0,5.0,80):
#     #calculating constants for the equation ay^2 + by+c
#     c = -0.5*(const + 1.0 + inv_coVar[0][0]*x*x - 2*wk[0]*x)
#     b = wk[1] - inv_coVar[1][0]*x
#     a = -0.5 * inv_coVar[1][1]
#     y=np.roots([a,b,c])
#     plt.plot(x,y[1],marker='x',c='black')
#     plt.plot(x,y[0],marker='x',c='black')
# plt.show()
    
