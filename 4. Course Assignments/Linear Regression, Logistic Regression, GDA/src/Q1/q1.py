import sys
import os
# from matplotlib import animation, markers
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# import threading
# import time

#reading the commandline arguments
trainDataDir=sys.argv[1]
testDataDir=sys.argv[2]

#reading independent variables from X.csv and dependent variable from Y.csv and normalizing them
X=pd.read_csv(os.path.join(trainDataDir,"X.csv"),header=None)
X=((X-X.mean())/X.std())
Y=pd.read_csv(os.path.join(trainDataDir,"Y.csv"),header=None)

#setting fig twice as wide as it is tall
# fig = plt.figure(figsize=(9,9))

#l is storing list of hystorical coordinate values of  (θ₀ as x), (θ₁ as y) and (J(θ) as z) for plotting live data
# l={'x':[],'y':[],'z':[]}

# keeping track of number of iterations
# iterations=0


#Implementation of Gredient Descent algorithm for linear regression
def GradientDescent(X,Y,thetas=np.zeros(2),ita=0.001,JTheta=0):
    # global iterations, l
    # iterations+=1
    # time.sleep(3)
    while True:
        # computing the hypothesis h(θ)
        calculatedOutput=thetas[0]+thetas[1]*X.iloc[:,0]

        preJTheta=JTheta

        # calculating h(θ)-Y and storing in temporary variable 
        temp=calculatedOutput-Y.iloc[:,0]
        # now calcumlating the J(θ) = (1/2*m)(h(θ)-Y)²
        JTheta=pow(temp,2).mean()/2

        #Stopping condition
        if(abs(JTheta-preJTheta)<0.00000001):
            break 
        
        #saving hystorical data for plotting graphs
        # l['x'].append(thetas[0])
        # l['y'].append(thetas[1])
        # l['z'].append(JTheta) 

        #applying gradient descent: updating parameters
        thetas[0]=thetas[0]-ita*temp.mean()
        thetas[1]=thetas[1]-ita*(temp*X.iloc[:,0]).mean()
    return (thetas, JTheta)
        
#training data
thetas, JTheta = GradientDescent(X,Y)
# thetas represent parameters in the hypothesis h(θ) = θ₀x₀ + θ₁x₁, initialize to 0
# JTheta is the error during convergence


# threading.Thread(target=GredientDescent).start()
# LIVE GRAPHS

# # Graph 1:3d meshgraph
# ax1=fig.add_subplot(2,2,1,projection='3d')
# spacing=50
# x=np.outer(np.linspace(-2,2,spacing),np.ones(spacing))
# y=x.copy().T
# z=np.ones((spacing,spacing))
# for i in range(spacing):
#     for j in range(spacing):
#         co=x[i,j]+y[i,j]*X.iloc[:,0]
#         t=co-Y.iloc[:,0]
#         z[i,j]=pow(t,2).mean()/2

# ax1.set_ylabel('θ₀')
# ax1.set_xlabel('θ₁')
# ax1.set_zlabel('J(θ)')
# ax1.plot_surface(x,y,z,cmap='viridis',alpha=0.4)
# # for animating meshgraph
# def animate(i):
#     global l
#     ax1.plot(l['x'],l['y'],l['z'], 'black', marker='.')

# #  Graph 2: plot the data and hypothesis graph
# ax2=fig.add_subplot(2,2,2)
# # for animating line graph
# def animate2(i):
#     global X,Y, thetas
#     ax2.clear()
#     ax2.grid()
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('Y')    
#     ax2.set_xlim([X.min()[0],X.max()[0]])
#     ax2.set_ylim([Y.min()[0],Y.max()[0]])
#     # ax2.legend(['Line: '+str(thetas[0])+' + '+str(thetas[1])+'x'])
#     # ax2.text(-1,3, 'Line: '+str(thetas[0])+' + '+str(thetas[1])+'x')
#     ax2.scatter(X,Y)
#     xs=np.array([X.min()[0],X.max()[0]])
#     ys=thetas[0]+thetas[1]*xs
#     ax2.plot(xs,ys,'black')


# #Graph 3: plot contour graph
# ax3=fig.add_subplot(2,2,3)
# cm=plt.contour(x,y,z)
# ax3.clabel(cm,inline=1, fontsize=10)
# ax3.set_ylabel('θ₀')
# ax3.set_xlabel('θ₁')
# #for animating contour graph
# def animate3(i):
#     # ax3.clear()
#     ax3.plot(l['x'],l['y'], marker='x',color='black')

# #Graph 4: value of J(θ) with increase in iteration
# ax4=fig.add_subplot(2,2,4)
# ax4.set_ylabel('J(θ)')
# ax4.set_xlabel('Iterations')
# def animate4(i):
#     global fig
#     if(len(l['z'])>0):
#         ax4.legend(['J(θ): '+str(l['z'][-1]),'Iterations: '+str(len(l['z']))])
#     ax4.plot(l['z'], 'black', marker='.')
#     fig.savefig('animation/fig'+str(i)+"png")


# ani=animation.FuncAnimation(fig,animate)
# ani2=animation.FuncAnimation(fig,animate2)
# ani3=animation.FuncAnimation(fig,animate3)
# ani4=animation.FuncAnimation(fig,animate4)
# plt.tight_layout()
# plt.show()

# # ani.save('animation.gif', writer=PillowWriter(fps=60))
# print([thetas, JTheta])



#testing data
XTest=pd.read_csv(testDataDir+"/X.csv",header=None)
XTest=((XTest-XTest.mean())/XTest.std())
YTest=thetas[0]+thetas[1]*XTest

pd.DataFrame(YTest).to_csv('result_1.txt',index=False,header=None)
    
