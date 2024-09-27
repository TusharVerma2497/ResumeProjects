import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
# This is working one somehow

class nn:
    def __init__(self, batchSize=100, feature_layer=None, hidden_layer=list(), output_layer=None, eta=0.1, activationFunctions=dict()):

        l=[self.sigmoid]*(len(hidden_layer)+1)
        dl=[self.sigmoidDerivative]*(len(hidden_layer)+1)
        for i in activationFunctions.keys():
            if(str.lower(activationFunctions[i])=='relu'):
                l[i-1]=self.ReLU
                dl[i-1]=self.ReLUDerivative
            elif(str.lower(activationFunctions[i])=='sigmoid'):
                pass

        self.activationFunction_=l
        self.activationDerivative_=dl
        self.batchSize_=batchSize
        self.layers=[feature_layer]+hidden_layer+[output_layer]
        self.eta=eta
        self.init_eta_=self.eta
        self.weights_=list()
        self.deltaW_=list()
        self.activations_=list()
        self.neuronInputs_=list()
        self.validationAccuracy_=-1
        self.validation_set_=None

        # initialize weights and biases
        l=self.layers
        for i in range((len(l)-1)):
            # self.weights_.append(np.random.rand(l[i]+1, l[i+1]))
            self.weights_.append(np.random.normal(size=(l[i]+1, l[i+1]),scale=1/np.sqrt(l[i])))
            self.deltaW_.append(np.zeros((l[i]+1, l[i+1])))
            self.activations_.append(np.zeros(l[i]))
            self.neuronInputs_.append(np.zeros(l[i]))
            #initializing bias to zeros
        for i in self.weights_:
            i[i.shape[0]-1]=np.zeros(i.shape[1])

    def resetDeltaW(self):
        self.deltaW_=list()
        l=self.layers
        for i in range((len(l)-1)):
            self.deltaW_.append(np.zeros((l[i]+1, l[i+1])))
    
    def sigmoid(self, l):
        x=l
        # return 1.0 / (1.0 + np.exp(-x))
        for i in range(len(x)):
            if x[i]<0:
                x[i]=np.exp(x[i]) / (1.0 + np.exp(x[i]))
            else:
                x[i]=1.0 / (1.0 + np.exp(-x[i]))
        return x

    def sigmoidDerivative(self,x):
        return x*(1.0 - x)

    def ReLU(self,x):
        return x*(x>0)
    
    def ReLUDerivative(self,l):
        x=l
        # print(l)
        # print('----------------------------------')
        for i in range(len(x)):
            if x[i]==0.0:
                x[i]=random.random()
                # x[i]=0.0
            elif x[i]>0.0:
                x[i]=1.0
            else:
                x[i]=0.0
        # print(x)
        return x

    def RMS(self,x,y):
        return np.average((y-x)**2)
    
    def RMSderivative(self,x,y):
        return (y-x)
    
    def feedForward(self,inputs):
        layer_output=list()
        layer_input=list()
        layer_output.append(inputs)
        layer_input.append(inputs)
        #activations from the first layer is the input
        activations=inputs
        for i in zip(self.weights_, self.activationFunction_):
            # calculating input to the neurons
            neuron_input=np.dot(np.append(activations,1),i[0])
            # neuron_input=np.matmul(np.append(activations,1).T,i[0])
            # passing it through the activation function for the given layer
            neuron_output=i[1](neuron_input)
            # activations for next layer is inputs from previous layer
            activations=neuron_output
            # saving the activations from each layers
            layer_output.append(neuron_output)
            layer_input.append(neuron_input)
        self.activations_=layer_output 
        self.neuronInputs_=layer_input
        return activations
    
    #adaptive learning
    def updateLearningRate(self,epoch):
        self.eta=self.init_eta_/np.sqrt(epoch)
    
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
        mask=[random.randint(0,len(inputs)-1) for i in range(4000)]
        self.validation_set_=[(inputs[i],targets[i]) for i in mask]

        epoch=1
        while(self.converge()!=True):
            # epoch
            self.updateLearningRate(epoch)
            for iter in range(int(len(targets)/self.batchSize_)):
                sumError=0
                #batch
                # after each bach reset the deltaW to 0s
                self.resetDeltaW()
                for i in range(self.batchSize_):
                    index=random.randint(0,len(inputs)-1)
                    o=self.feedForward(inputs[index])
                    error=self.RMSderivative(o,targets[index])
                    self.backPropogation(error)
                    sumError += self.RMS(o,targets[index])
                self.gradientDescent()    
            print('epoch: {}  RMS: {}'.format(epoch,sumError))
            print('learning rate used: {}'.format(self.eta))
            epoch+=1
            


    def backPropogation(self,error):
        # moving backwards
        for i in range(len(self.weights_)-1,-1,-1):
            delW=error*self.activationDerivative_[i](self.activations_[i+1])
            # if(type(self.activationFunction_[i])==type(self.sigmoid)):
            #     delW=error*self.activationDerivative_[i](self.activations_[i+1])
            # else:
            #     delW=error*self.activationDerivative_[i](self.neuronInputs_[i+1])
            act=(np.append(self.activations_[i],1)).reshape(self.layers[i]+1,1)
            delW=delW.reshape(1,self.layers[i+1])
            # calculating deltaW for weights and bias of a layer
            self.deltaW_[i]+=np.matmul(act,delW)
            # calculating error for the previous layer
            error=np.dot(delW,self.weights_[i].T)
            # dropping error for bias not, no need to backpropogate it has no back connections
            error=np.delete(error,error.shape[1]-1)


    def gradientDescent(self):
        for i in range(len(self.weights_)):
            self.weights_[i]+=(self.deltaW_[i]/self.batchSize_)*self.eta
            # self.weights_[i]+=self.deltaW_[i]*self.eta


file=open(sys.argv[3]+'/e.txt', 'w')
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
    # data=data/255.0
    return (data,output)


def test(inputs,targets):
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
    file.write("\n------------------------------------------------\n\n")
    #calculating the accuracy
    accuracy=np.trace(confusionMatrix)*100/np.sum(confusionMatrix)
    print('Accuracy: {}%'.format(accuracy))
    file.write('Accuracy: {}%\n'.format(accuracy))
    file.flush()
    return accuracy


train_data=None
test_data=None
hidden_layers=[2,3,4,5]
# setting number of hidden nodes for each layer
hidden_nodes=[[50]*i for i in hidden_layers]
# setting the activation functions for each layer
activationFunctions=[[{1:'relu'},
                    {1:'relu', 2:'relu'},
                    {1:'relu', 2:'relu',3:'relu'}, 
                    {1:'relu', 2:'relu',3:'relu',4:'relu'}],
                    [dict(),dict(),dict(),dict()]]
maxAccuracy=-1
index=-1
bestArch=dict()
# looping through one for ReLU and two for sigmoid
for i in range(2):
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
        mod=nn(feature_layer=784,hidden_layer=hidden_nodes[hidden],
            output_layer=10,batchSize=100,activationFunctions=activationFunctions[i][hidden])
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

    # Graph for accuracy vs number of hidden layers
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(hidden_layers, accuracy[0], marker="o")
    ax1.plot(hidden_layers, accuracy[1], marker="o")
    ax1.set_xlabel("#Hidden_layers of 50 nodes each")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy for training and test data")
    ax1.legend(["training", "test"])

    # Graph for accuracy vs number of hidden nodes
    ax2=fig.add_subplot(2,1,2)
    ax2.plot(hidden_layers, time_taken, marker="o")
    ax2.set_xlabel("#Hidden_layers with 50 nodes each")
    ax2.set_ylabel("Time(sec)")
    ax2.set_title("Time taken by training and test data")
    ax2.legend(["training time"])

    plt.savefig(sys.argv[3]+'/e_Graphs'+str(i))
    plt.tight_layout()
    print('All the asked plots are saved as Graphs.png')
    # plt.show()

    # finding and saving the best model based on test accuracy
    if(maxAccuracy< np.max(accuracy[1])):
        index=accuracy[1].index(np.max(accuracy[1]))
        bestArch=activationFunctions[i][index]

with open('e_bestArchitecture','wb') as f:
    pickle.dump((hidden_nodes[index],bestArch),f)


file.close()


# All For ReLU
# Validation accuracy: 83%
# Time taken to train: 687.1628496646881
# model [50, 50] hidden layers
# ----------------confusion matrix----------------
# [[5156   11  103  275   14    7 1380    0   16    1]
#  [  35 5709    7   78   17    2   23    0    5    1]
#  [  84   74 4446   54  480    3  703    0   26    0]
#  [ 380  169   61 5227  250    3  220    0   52    1]
#  [  25   14  851  206 4634    0  672    0   25    0]
#  [  18    1    9    2    8 5424    6  231   39  122]
#  [ 210   18  460  134  549    1 2866    0   87    0]
#  [   0    0    1    0    0  349    1 5435   32  287]
#  [  92    4   62   23   47   35  128   15 5706    2]
#  [   0    0    0    1    1  176    1  319   12 5585]]
# ------------------------------------------------
# Accuracy: 83.64806080101334%
# ----------------confusion matrix----------------
# [[845   5  21  41   0   1 229   0   2   0]
#  [  3 942   0  12   2   1   1   0   1   0]
#  [  9  11 727  11 102   0 129   0   5   0]
#  [ 66  34  12 860  42   1  54   0  10   0]
#  [  4   7 141  33 748   0 111   0   2   0]
#  [  4   0   3   2   0 880   3  38   8  22]
#  [ 51   0  83  35  99   0 447   0  16   1]
#  [  0   0   0   0   0  64   0 907   4  49]
#  [ 18   1  13   5   7   8  26   1 951   0]
#  [  0   0   0   1   0  45   0  54   1 927]]
# ------------------------------------------------
# Accuracy: 82.34823482348234%

# Validation accuracy: 84%
# Time taken to train: 949.8115842342377
# model [50, 50, 50] hidden layers
# ----------------confusion matrix----------------
# [[5095    6   85  240    5    1 1076    0   16    0]
#  [  27 5740   11   90   15    2   20    0    3    3]
#  [  76   77 4515   57  565    3  755    0   21    0]
#  [ 337  140   56 5204  220    5  203    0   36    1]
#  [  17   16  749  213 4578    0  620    0   19    0]
#  [   8    2   10    1    2 5522    4  200   39  119]
#  [ 362   15  525  165  580    0 3202    0   94    1]
#  [   0    1    0    1    0  304    2 5545   25  308]
#  [  77    3   49   27   34   31  117   13 5742    2]
#  [   1    0    0    2    1  132    1  242    5 5565]]
# ------------------------------------------------
# Accuracy: 84.51474191236521%
# ----------------confusion matrix----------------
# [[822   0  20  31   0   0 175   0   3   0]
#  [  2 950   3  18   2   0   2   0   1   0]
#  [  7  11 732   8 114   0 131   0   5   0]
#  [ 63  31  11 850  39   2  49   0   4   0]
#  [  4   5 130  33 744   0  93   0   5   0]
#  [  2   0   2   2   0 900   1  40   9  19]
#  [ 83   1  94  53  96   0 523   0  16   0]
#  [  0   0   0   0   0  58   1 922   3  56]
#  [ 16   2   8   4   5   8  25   1 953   1]
#  [  1   0   0   1   0  32   0  37   1 923]]
# ------------------------------------------------
# Accuracy: 83.1983198319832%


# Validation accuracy: 84%
# Time taken to train: 807.3541448116302
# model [50, 50, 50, 50] hidden layers
# ----------------confusion matrix----------------
# [[5004   12   77  240    7    6 1248    0   17    1]
#  [   4 5684    5   46    5    1   10    0    2    4]
#  [ 107   85 4558   41  532    1  679    0   27    0]
#  [ 428  175   64 5287  236    6  253    0   70    0]
#  [  18   16  742  231 4651    0  629    0   35    0]
#  [  11    1    8    1    8 5452    3  194   45   98]
#  [ 341   18  505  127  524    3 3075    0   94    0]
#  [   2    1    0    2    0  316    4 5484   27  220]
#  [  82    8   41   24   36   34   97   17 5674    2]
#  [   3    0    0    1    1  181    2  305    9 5674]]
# ------------------------------------------------
# Accuracy: 84.23973732895548%
# ----------------confusion matrix----------------
# [[810   1  10  42   0   1 214   0   1   1]
#  [  0 944   1   4   0   1   1   0   1   0]
#  [ 12  12 743  12 111   0 128   0  10   0]
#  [ 78  35  13 865  38   0  49   0   9   0]
#  [  4   5 130  33 751   0 102   0   5   0]
#  [  3   0   3   2   1 883   0  33   9  17]
#  [ 78   1  94  35  92   0 477   0  17   1]
#  [  0   0   1   1   1  62   1 915   4  43]
#  [ 15   2   5   5   6   8  28   1 942   0]
#  [  0   0   0   1   0  45   0  51   2 937]]
# ------------------------------------------------
# Accuracy: 82.67826782678267%


# Validation accuracy: 84%
# Time taken to train: 971.501805305481
# model [50, 50, 50, 50, 50] hidden layers
# ----------------confusion matrix----------------
# [[5031   11   73  214    8    1 1063    0   15    0]
#  [  11 5687    3   52   14    5    9    0   15    4]
#  [  84   56 4470   53  535    1  731    0   21    0]
#  [ 342  153   63 5300  242    1  233    0   54    0]
#  [  15   47  709  179 4514    0  511    0   28    0]
#  [   8    3   13    0    3 5463    5  112   23   94]
#  [ 417   40  633  179  653    1 3338    0  104    0]
#  [   0    0    0    0    0  357    2 5614   27  260]
#  [  81    3   36   20   30   28  105   15 5703    4]
#  [  11    0    0    3    1  143    3  259   10 5637]]
# ------------------------------------------------
# Accuracy: 84.59640994016567%
# ----------------confusion matrix----------------
# [[809   2  16  35   0   0 175   0   0   0]
#  [  1 944   1  11   1   1   4   0   3   0]
#  [ 12  11 725   9 109   0 121   0   3   0]
#  [ 67  29  15 856  40   0  49   0   8   0]
#  [  5  10 124  30 738   0  94   0   3   0]
#  [  2   0   2   1   0 884   0  21   7  14]
#  [ 89   3 111  53 106   0 533   0  20   2]
#  [  0   0   0   0   0  71   1 940   5  52]
#  [ 14   1   6   5   6   8  22   1 950   0]
#  [  1   0   0   0   0  36   1  38   1 931]]
# ------------------------------------------------
# Accuracy: 83.1083108310831%


# Sigmoid
# Validation accuracy: 73%
# Time taken to train: 1145.8908560276031
# model [50, 50] hidden layers
# ----------------confusion matrix----------------
# [[5123   32  358  626  156    0 1935    0   72    0]
#  [  62 5661   43  148   65    6   32    0    8    4]
#  [  53   90 3647   42  872    0 1247    0   34    0]
#  [ 513  191   55 4931  601    1  297    0   31    2]
#  [  36    8 1474  130 4073    0 1537    0   17    0]
#  [  44    9   73   59   31 4552  101  474   75  117]
#  [  18    4  229   32  129    0  588    0   11    0]
#  [   1    1    0    0    0 1041    1 5004   46  280]
#  [ 148    4  120   30   72   69  261   11 5685    3]
#  [   2    0    1    2    1  331    1  511   21 5593]]
# ------------------------------------------------
# Accuracy: 74.76291271521193%
# ----------------confusion matrix----------------
# [[846   9  59 124  16   0 311   0  15   0]
#  [  9 932   9  19  10   1   6   0   2   0]
#  [  5  14 610   7 171   0 210   0   9   0]
#  [ 91  36  12 793  87   1  67   0   5   0]
#  [ 11   6 253  25 674   0 244   0   3   0]
#  [ 10   1   7  16   7 758  18  65  19  25]
#  [  5   0  27   9  25   0  96   0   1   0]
#  [  0   0   0   0   0 167   0 845   7  42]
#  [ 23   2  23   6  10  10  48   1 937   1]
#  [  0   0   0   1   0  63   0  89   2 931]]
# ------------------------------------------------
# Accuracy: 74.22742274227423%

# Validation accuracy: 47%
# Time taken to train: 2317.100964307785
# model [50, 50, 50] hidden layers
# ----------------confusion matrix----------------
# [[   0    0    0    0    0    0    0    0    0    0]
#  [1666 5778  612 4134  503   43 1256    0  348   36]
#  [ 175   42 2266   22 1488    2 1236    0   96   13]
#  [1810   91  129 1451  470    1  738    0   15    1]
#  [2215   80 2780  376 3472    0 2457    0   33    5]
#  [  13    1    9    1    3   81    9    1   89   15]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   4    1    4    0    1 5343   14 5757  341  929]
#  [ 116    7  197   16   62   12  288    0 4897   26]
#  [   1    0    3    0    1  518    2  242  181 4974]]
# ------------------------------------------------
# Accuracy: 47.79412990216504%
# ----------------confusion matrix----------------
# [[  0   0   0   0   0   0   0   0   0   0]
#  [258 958  91 682  77   6 200   0  62   5]
#  [ 30   8 396   6 264   0 203   0  20   2]
#  [324  14  25 232  70   0 130   0   2   0]
#  [365  17 453  75 576   0 403   0   5   3]
#  [  3   1   0   1   0  13   4   0  19   1]
#  [  0   0   0   0   0   0   0   0   0   0]
#  [  2   0   0   0   0 890   1 963  52 160]
#  [ 16   2  34   3  13   0  59   0 820   5]
#  [  2   0   1   1   0  91   0  37  20 823]]
# ------------------------------------------------
# Accuracy: 47.814781478147815%


# Validation accuracy: 27%
# Time taken to train: 2650.133682012558
# model [50, 50, 50, 50] hidden layers
# ----------------confusion matrix----------------
# [[4604 2570  152 4726  972    0 1500    0  120    0]
#  [   1 1849    0    4    0    0    1    0    0    0]
#  [   2    0    0    0    0    0    0    0    2    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [ 233  728 3624  103 3838    2 1961    0  126   10]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [1160  853 2224 1167 1190 5998 2538 6000 5752 5989]]
# ------------------------------------------------
# Accuracy: 27.13378556309272%
# ----------------confusion matrix----------------
# [[ 765  455   31  785  136    0  255    0   24    0]
#  [   0  309    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    1    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [  40   96  589   22  659    0  311    0   34    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [ 195  140  380  193  205 1000  434 1000  941  999]]
# ------------------------------------------------
# Accuracy: 27.322732273227324%


# Time taken to train: 2852.7096073627472
# model [50, 50, 50, 50, 50] hidden layers
# ----------------confusion matrix----------------
# [[   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [6000 6000 6000 6000 6000 5944 6000 5999 5979 5968]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0   56    0    1   21   31]]
# ------------------------------------------------
# Accuracy: 10.051834197236621%
# ----------------confusion matrix----------------
# [[   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [1000 1000 1000 1000 1000  996 1000 1000  997  997]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    0    0    0    0    0]
#  [   0    0    0    0    0    4    0    0    3    2]]
# ------------------------------------------------
# Accuracy: 10.021002100210021%