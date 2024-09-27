import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys

# This is working one

class nn:
    def __init__(self, batchSize=100, feature_layer=None, hidden_layer=list(), output_layer=None, eta=0.1, activationFunctions=dict()):

        l=[self.sigmoid]*(len(hidden_layer)+1)
        dl=[self.sigmoidDerivative]*(len(hidden_layer)+1)
        for i in activationFunctions.keys():
            if(str.lower(activationFunctions[i])=='softmax'):
                l[i-1]=self.softMax
                dl[i-1]=self.softMaxDerivative
            elif(str.lower(activationFunctions[i])=='sigmoid'):
                pass

        self.activationFunction_=l
        self.activationDerivative_=dl
        self.batchSize_=batchSize
        self.layers=[feature_layer]+hidden_layer+[output_layer]
        self.eta=eta
        self.weights_=list()
        self.deltaW_=list()
        self.activations_=list()
        self.predeltaW_=None
        self.validationAccuracy_=0
        self.validation_set_=None

        # initialize weights and biases
        l=self.layers
        for i in range((len(l)-1)):
            self.weights_.append(np.random.rand(l[i]+1, l[i+1]))
            self.deltaW_.append(np.zeros((l[i]+1, l[i+1])))
            self.activations_.append(np.zeros(l[i]))

    def resetDeltaW(self):
        self.deltaW_=list()
        l=self.layers
        for i in range((len(l)-1)):
            self.deltaW_.append(np.zeros((l[i]+1, l[i+1])))
      
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoidDerivative(self,x):
        return x*(1.0 - x)

    def RMS(self,x,y):
        return np.average((y-x)**2)
    
    def RMSderivative(self,x,y):
        return 2*(x-y)
    
    def feedForward(self,inputs):
        layer_output=list()
        layer_output.append(inputs)
        #activations from the first layer is the input
        activations=inputs
        for i in zip(self.weights_, self.activationFunction_):
            # calculating input to the neurons
            neuron_input=np.dot(np.append(activations,1),i[0])
            # passing it through the activation function for the given layer
            neuron_output=i[1](neuron_input)
            # activations for next layer is inputs from previous layer
            activations=neuron_output
            # saving the activations from each layers
            layer_output.append(neuron_output)
        self.activations_=layer_output 
        return activations
    
    def converge(self):

        mask=np.array([0,1,2,3,4,5,6,7,8,9])
        correct=0
        for v in self.validation_set_:
            modelOutput=self.feedForward(v[0])
            # class with the maximum activation is our predicted class
            modelOutput=np.array([1 if i==max(modelOutput) else 0 for i in modelOutput])
            # creating a class mask for convinience in confusion matrix
            modelOutput=np.dot(mask,modelOutput)
            targetOutput=int(np.dot(mask,v[1]))
            if modelOutput==targetOutput:
                correct+=1
        accuracy=int(correct*100/len(self.validation_set_))
        print('Validation accuracy: {}%'.format(accuracy))
        if(accuracy==self.validationAccuracy_):
            return True
        elif(self.validationAccuracy_<accuracy):
            self.validationAccuracy_=accuracy
        return False

    def train(self,inputs,targets):

        # creating a validation set which will be used in checking convergence
        mask=[random.randint(0,len(inputs)-1) for i in range(5000)]
        self.validation_set_=[(inputs[i],targets[i]) for i in mask]

        epoch=1
        while(self.converge()!=True):
            # epoch
            for iter in range(int(len(targets)/self.batchSize_)):
                sumError=0
                self.predeltaW_=self.deltaW_
                #batch
                # after each bach reset the deltaW
                self.resetDeltaW()
                for i in range(self.batchSize_):
                    index=random.randint(0,len(inputs)-1)
                    o=self.feedForward(inputs[index])
                    error=self.RMSderivative(o,targets[index])
                    self.backPropogation(error)
                    sumError += self.RMS(o,targets[index])
                self.gradientDescent()    
            print('epoch: {}  RMS: {}'.format(epoch,sumError))
            epoch+=1
            


    def backPropogation(self,error):
        # moving backwards
        for i in range(len(self.weights_)-1,-1,-1):
            delW=error*self.activationDerivative_[i](self.activations_[i+1])
            act=(np.append(self.activations_[i],1)).reshape(self.layers[i]+1,1)
            delW=delW.reshape(1,self.layers[i+1])
            self.deltaW_[i]+=np.matmul(act,delW)
            error=np.dot(delW,self.weights_[i].T)
            error=np.delete(error,error.shape[1]-1)


    def gradientDescent(self):
        for i in range(len(self.weights_)):
            self.weights_[i]-=self.deltaW_[i]*self.eta


file=open(sys.argv[3]+'/b.txt', 'w')
ohe=OneHotEncoder()
def dataPreperation(data):
    # encoding output as onehotEncoding
    output=ohe.fit_transform(data[['9']])
    output=np.array(output.toarray())
    #removing the output column from the dataframe
    data=data.drop(['9'],axis=1)

    # normalizing the pixels
    mean=np.mean(data, axis=0)
    std=np.std(data,axis=0)
    data=(data-mean)/std
    data=data.to_numpy()
    return (data,output)


def test(inputs,targets):
    global file
    # creating a class mask for convinience in confusion matrix
    mask=np.array([0,1,2,3,4,5,6,7,8,9])
    #initializing the confusion matrix
    confusionMatrix=np.zeros((10,10),dtype=int)
    for i in range(len(targets)):
        modelOutput=mod.feedForward(inputs[i])
        # class with the maximum activation is our predicted class
        modelOutput=np.array([1 if i==max(modelOutput) else 0 for i in modelOutput])
        
        modelOutput=np.dot(mask,modelOutput)
        targetOutput=int(np.dot(mask,targets[i]))
        confusionMatrix[modelOutput][targetOutput]+=1
    # # modelOutput=[round(k) for k in modelOutput]
    print("----------------confusion matrix----------------")
    file.write("----------------confusion matrix----------------\n")
    print(confusionMatrix[:10])
    file.write(str(confusionMatrix[:10]))
    print("------------------------------------------------")
    file.write("\n------------------------------------------------\n")
    #calculating the accuracy
    accuracy=np.trace(confusionMatrix)*100/np.sum(confusionMatrix)
    print('Accuracy: {}%'.format(accuracy))
    file.write('Accuracy: {}%\n'.format(accuracy))
    file.flush()
    return accuracy


train_data=None
test_data=None
hidden_nodes=[5,10,15,20,25]
# hidden_nodes=[2,3]

accuracy=np.zeros((2,len(hidden_nodes)))
# row one will be the training accuracies
# column two will be test accuracies

time_taken=np.zeros(len(hidden_nodes))
# will store the training time


for hidden in range(len(hidden_nodes)):
    
    if(type(train_data)==type(None)):
        # loading training data
        with open(sys.argv[1],'r') as f:
            train_data=pd.read_csv(f)
        # loading test data
        with open(sys.argv[2],'r') as f:
            test_data=pd.read_csv(f)

        train_data,train_output=dataPreperation(train_data)
        test_data,test_output=dataPreperation(test_data)

    # Training the model
    st=time.time()
    mod=nn(feature_layer=784,hidden_layer=[hidden_nodes[hidden]],output_layer=10,batchSize=1)
    mod.train(train_data,train_output)
    time_taken[hidden]=time.time()-st

    print("Time taken to train: {}".format(time_taken[hidden]))
    file.write("Time taken to train: {}\n".format(time_taken[hidden]))

    # # saving model
    # with open('model'+str(hidden),'wb') as f:
    #     pickle.dump(mod,f)

    # # loading model
    # with open('model5','rb') as f:
    #     mod=pickle.load(f)


    print('model {} hidden layers'.format(hidden_nodes[hidden]))
    file.write('model {} hidden layers\n'.format(hidden_nodes[hidden]))
    # Training accuracy
    accuracy[0][hidden]=test(train_data,train_output)
    # Test accuracy
    accuracy[1][hidden]=test(test_data,test_output)


# plotting the graphs
fig = plt.figure(figsize=(7,12))

# Graph for accuracy vs number of hidden nodes
ax1=fig.add_subplot(2,1,1)
ax1.plot(hidden_nodes, accuracy[0], marker="o")
ax1.plot(hidden_nodes, accuracy[1], marker="o")
ax1.set_xlabel("#Hidden_nodes")
ax1.set_ylabel("Accuracy")
ax1.set_title("Accuracy for training and test data")
ax1.legend(["training", "test"])

# Graph for accuracy vs number of hidden nodes
ax2=fig.add_subplot(2,1,2)
ax2.plot(hidden_nodes, time_taken, marker="o")
ax2.set_xlabel("#Hidden_nodes")
ax2.set_ylabel("Time(sec)")
ax2.set_title("Time taken by training and test data")
ax2.legend(["training time"])

plt.savefig(sys.argv[3]+'/b_Graphs')
plt.tight_layout()
print('All the asked plots are saved as Graphs.png')
# plt.show()


file.close()









# Validation accuracy: 10%
# epoch: 1  RMS: 0.003044176035200955
# Validation accuracy: 56%
# epoch: 2  RMS: 0.04380509930455249
# Validation accuracy: 72%
# epoch: 3  RMS: 0.12626024646678563
# Validation accuracy: 77%
# epoch: 4  RMS: 0.0001554851844095746
# Validation accuracy: 77%
# Time taken to train: 37.8497838973999
# model 0 hidden layers
# ----------------confusion matrix----------------
# [[4980   48  119  630   98   17 1254    0   24    5]
#  [   8 5496    8   45   12    2   12    0    9    0]
#  [  57  111 3108   31  315    0  535    0   41    1]
#  [ 212  249   66 4508  183    0  205    0   14    0]
#  [ 143   78 1621  604 4403    3  714    0   39   10]
#  [   7    0    5    0    0 5374    4  983   26  227]
#  [ 412   10  970  139  926    2 3019    0   89    0]
#  [   0    0    2    0    2  298    2 4849   52  512]
#  [ 177    6   99   42   60   91  250   20 5652    4]
#  [   4    2    2    1    1  213    5  148   54 5240]]
# ------------------------------------------------
# Accuracy: 77.71629527158785%
# ----------------confusion matrix----------------
# [[791  14  26 105  12   3 201   0   2   0]
#  [  2 904   0   8   0   0   0   0   1   0]
#  [ 10  12 503   5  70   0 106   0   9   0]
#  [ 37  50  12 741  22   0  42   0   2   0]
#  [ 33  15 276 104 724   0 119   0   8   2]
#  [  2   0   1   1   0 887   0 140   6  43]
#  [ 92   4 168  28 160   0 488   0  11   0]
#  [  0   0   0   0   0  57   0 835   6  96]
#  [ 32   1  13   8  11  13  44   1 946   1]
#  [  1   0   1   0   1  40   0  24   9 857]]
# ------------------------------------------------
# Accuracy: 76.76767676767676%
# Validation accuracy: 7%
# epoch: 1  RMS: 0.054751612813150705
# Validation accuracy: 59%
# epoch: 2  RMS: 0.00038656893379471446
# Validation accuracy: 74%
# epoch: 3  RMS: 0.0005365790105743425
# Validation accuracy: 77%
# epoch: 4  RMS: 0.14444921626470947
# Validation accuracy: 77%
# Time taken to train: 40.40649962425232
# model 1 hidden layers
# ----------------confusion matrix----------------
# [[5172   50  191  447   69    1 1599    0   35    2]
#  [  32 5709   10   72   18    5   17    0    8    3]
#  [ 101   62 3719   38  268    5  725    0  112    0]
#  [ 391  133   95 5024  291    3  307    0   27    0]
#  [  87   26 1678  222 4985    3 2441    0   44    1]
#  [   5    1    3    1    2 5108    1  130   49  148]
#  [ 103   15  242  167  330    0  761    0   10    0]
#  [   2    0    0    0    0  520    1 5665   35  495]
#  [ 105    3   62   28   36   97  146   20 5660   22]
#  [   2    1    0    1    1  258    2  185   20 5328]]
# ------------------------------------------------
# Accuracy: 78.55297588293138%
# ----------------confusion matrix----------------
# [[838  12  34  65   7   0 259   0  12   0]
#  [  5 941   1  12   5   1   2   0   0   0]
#  [ 17  10 581   7  66   0 121   0  19   1]
#  [ 82  25  19 837  47   0  60   0   3   0]
#  [ 11   8 309  38 824   0 395   0   7   0]
#  [  2   0   1   1   1 839   1  25  12  18]
#  [ 28   3  44  33  44   0 130   0   2   0]
#  [  0   0   0   0   0  92   0 942   7  79]
#  [ 17   1  11   7   6  14  32   0 935   3]
#  [  0   0   0   0   0  54   0  33   3 898]]
# ------------------------------------------------
# Accuracy: 77.65776577657766%
# Validation accuracy: 12%
# epoch: 1  RMS: 0.11277478639384897
# Validation accuracy: 62%
# epoch: 2  RMS: 0.014553133615927768
# Validation accuracy: 76%
# epoch: 3  RMS: 0.17774246724301107
# Validation accuracy: 78%
# epoch: 4  RMS: 0.054066655163468536
# Validation accuracy: 80%
# epoch: 5  RMS: 0.051645549365220825
# Validation accuracy: 82%
# epoch: 6  RMS: 0.0010247790339079944
# Validation accuracy: 81%
# /home/tusharverma/MEGA/MEGAsync/MSR at IITD/First Sem/Courses/Machine Learning COL774/Assignments/Assignment 3/Q2/qa/qa.py:49: RuntimeWarning: overflow encountered in exp
#   return 1.0 / (1.0 + np.exp(-x))
# epoch: 7  RMS: 0.0010485571167523197
# Validation accuracy: 83%
# epoch: 8  RMS: 0.00044948234046213246
# Validation accuracy: 83%
# Time taken to train: 305.5929822921753
# model 2 hidden layers
# ----------------confusion matrix----------------
# [[4555   27   41  231   17    1  750    0   14    0]
#  [  45 5640   11   48   13    7   18    0    3    1]
#  [  79  112 3888   65  387    4  420    1   44    0]
#  [ 445  172   56 5190  153    5  281    0   63    0]
#  [  34   21 1168  275 4717    0  748    0   38    1]
#  [  13    2    5    3    2 5509    7  281   39  115]
#  [ 738   24  801  180  692    3 3688    0  128    0]
#  [   1    0    1    0    0  285    1 5500   21  292]
#  [  87    2   27    6   19   32   83   12 5644    2]
#  [   3    0    2    2    0  154    4  206    6 5588]]
# ------------------------------------------------
# Accuracy: 83.19971999533325%
# ----------------confusion matrix----------------
# [[719   3   6  34   1   0 127   0   2   0]
#  [  6 927   0   9   1   1   1   0   0   0]
#  [  9  15 627  11  72   0  76   0  11   0]
#  [ 84  41  15 850  24   2  62   0   9   1]
#  [  9   8 197  52 771   0 129   0   8   0]
#  [  2   0   0   0   0 906   1  48  10  13]
#  [159   5 148  39 123   0 582   0  26   1]
#  [  0   0   0   0   0  52   1 915   4  53]
#  [ 12   1   7   3   8   4  21   0 929   1]
#  [  0   0   0   2   0  35   0  37   1 930]]
# ------------------------------------------------
# Accuracy: 81.56815681568156%
# Validation accuracy: 7%
# epoch: 1  RMS: 0.011214869929128434
# Validation accuracy: 66%
# epoch: 2  RMS: 0.00141726407981914
# Validation accuracy: 73%
# epoch: 3  RMS: 0.10339243334374279
# Validation accuracy: 78%
# epoch: 4  RMS: 0.008795545828108866
# Validation accuracy: 79%
# epoch: 5  RMS: 0.00017786619508392815
# Validation accuracy: 80%
# epoch: 6  RMS: 0.0025381454947159396
# Validation accuracy: 81%
# epoch: 7  RMS: 0.012715676092946118
# Validation accuracy: 81%
# Time taken to train: 265.26688027381897
# model 3 hidden layers
# ----------------confusion matrix----------------
# [[4954   24  209  272   51    0 1376    0   58    1]
#  [  21 5692   17   85   17    0   20    0    3    0]
#  [ 130   83 4793   77 1271    2 1109    0   75    2]
#  [ 327  138   36 4964  201    2  205    0   14    2]
#  [ 113   38  718  271 3940    1  582    0   32    0]
#  [   3    2    3    7    1 5415    0  374   22  129]
#  [ 373   20  173  299  487    1 2561    0   27    0]
#  [   2    0    2    0    1  288    1 5230   16  239]
#  [  72    2   41   19   29   95  135   21 5716   10]
#  [   5    1    8    6    2  196   11  375   37 5616]]
# ------------------------------------------------
# Accuracy: 81.46969116151936%
# ----------------confusion matrix----------------
# [[810   5  45  51   9   0 233   0  12   0]
#  [  1 936   5  17   0   0   5   0   1   0]
#  [ 21  16 784  16 229   0 197   0  13   0]
#  [ 57  28  11 796  37   1  45   0   3   0]
#  [ 19   9 118  49 653   0 107   0   5   0]
#  [  2   0   0   2   0 894   1  69   6  30]
#  [ 73   6  25  64  64   0 384   0   4   0]
#  [  0   0   1   0   0  44   0 864   4  43]
#  [ 17   0   8   3   8  17  25   1 943   1]
#  [  0   0   3   2   0  44   3  66   9 925]]
# ------------------------------------------------
# Accuracy: 79.89798979897989%
# Validation accuracy: 12%
# epoch: 1  RMS: 0.10494700148449436
# Validation accuracy: 64%
# epoch: 2  RMS: 0.0009877641613797489
# Validation accuracy: 70%
# epoch: 3  RMS: 0.004158159997236189
# Validation accuracy: 78%
# epoch: 4  RMS: 0.00022973720900049685
# Validation accuracy: 79%
# epoch: 5  RMS: 0.0003360205405931947
# Validation accuracy: 81%
# epoch: 6  RMS: 0.05553832839224175
# Validation accuracy: 81%
# Time taken to train: 241.0044286251068
# model 4 hidden layers
# ----------------confusion matrix----------------
# [[4824   26   71  256   66    0 1099    0   18    0]
#  [  32 5630   10   91   28    0   27    0    2    2]
#  [ 357  123 5077   86 1480    3 1418    0  269    2]
#  [ 307  163   47 5087  260    1  183    0   21    0]
#  [  76   43  524  196 3746    1  532    1   37    0]
#  [   5    4    9   27    6 5503    5  302   40  149]
#  [ 351    8  231  233  378    1 2680    0   89    0]
#  [   0    1    1    1    0  194    1 5319   28  140]
#  [  47    2   30   22   35   51   54   14 5470    1]
#  [   1    0    0    1    1  246    1  364   26 5705]]
# ------------------------------------------------
# Accuracy: 81.73636227270454%
# ----------------confusion matrix----------------
# [[756   4  12  45  10   0 200   0   4   0]
#  [  3 933   2  19   5   1   5   0   1   0]
#  [ 78  15 842  12 277   0 246   0  56   1]
#  [ 57  32  12 818  39   0  39   0   5   0]
#  [ 19  13  82  43 585   0 105   0   6   0]
#  [  2   0   2   4   0 899   1  58   9  24]
#  [ 73   3  42  56  75   0 393   0  15   0]
#  [  0   0   0   0   0  44   0 886   5  35]
#  [ 11   0   5   2   9   9  11   0 892   0]
#  [  1   0   1   1   0  47   0  56   7 939]]
# ------------------------------------------------
# Accuracy: 79.43794379437944%