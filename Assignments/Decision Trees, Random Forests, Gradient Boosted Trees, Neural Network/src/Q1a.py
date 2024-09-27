import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import sys


def cleanData(data):
    # dropping the missing value
    data.replace({'?': np.nan},inplace =True)
    data=data.dropna(axis=0,how='any')
    # extracting input and output
    Y=np.array(data[['Severity']])
    X=data.drop(['BI-RADS assessment','Severity'],axis=1)
    return (X,Y)

def trainTree(X,Y):
    # Training the model
    print('Training start')
    model=tree.DecisionTreeClassifier()
    model=model.fit(X,Y)
    return model

def plotTree(model):
    plt.figure(figsize=(12, 9))
    tree.plot_tree(model,filled=True)
    plt.savefig(sys.argv[4]+'/1_a')
    # plt.show()
    
def accuracy(X,Y, model):
    Y=Y.reshape(len(Y))
    pre=model.predict(X)
    error=(Y-pre)**2
    return(100-(error.sum()*100/len(error)))


# Reading Training data
with open(sys.argv[1],'r') as f:
    train_data=pd.read_csv(f)

# Reading Test data
with open(sys.argv[3],'r') as f:
    test_data=pd.read_csv(f)

# Reading Validation data
with open(sys.argv[2],'r') as f:
    val_data=pd.read_csv(f)

file=open(sys.argv[4]+'/1_a.txt', 'w')

#cleaning train data
X,Y=cleanData(data=train_data)
#Training the model
modelTrain=trainTree(X=X,Y=Y)

#plotting the tree
plotTree(modelTrain)

#Train data accuracy
acc=accuracy(X,Y, modelTrain)
print("Accuracy of Training data: {}".format(acc))
file.write("Accuracy of Training data: {}\n".format(acc))

#Test data accuracy
#cleaning test data
X,Y=cleanData(data=test_data)
acc=accuracy(X,Y, modelTrain)
print("Accuracy of Test data: {}".format(acc))
file.write("Accuracy of Test data: {}\n".format(acc))

#Validation data accuracy
#cleaning validation data
X,Y=cleanData(data=val_data)
acc=accuracy(X,Y, modelTrain)
print("Accuracy of Validation data: {}".format(acc))
file.write("Accuracy of Validation data: {}\n".format(acc))

file.close()

# Training start
# Accuracy of Training data: 92.52747252747253
# Accuracy of Test data: 69.1699604743083
# Accuracy of Validation data: 76.03305785123968