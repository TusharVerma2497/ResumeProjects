import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import sys
import pickle


def cleanData(data):
    # dropping the missing value
    data.replace({'?': np.nan},inplace =True)
    data=data.dropna(axis=0,how='any')

    # extracting input and output
    Y=np.array(data[['Severity']])
    X=data.drop(['BI-RADS assessment','Severity'],axis=1)
    return (X,Y)

def trainTree(X,Y,alpha):
    # Training the model
    # print('Training start')
    model=tree.DecisionTreeClassifier(ccp_alpha=alpha)
    model=model.fit(X,Y)
    return model

def plotTree(model):
    plt.figure(figsize=(12, 9))
    tree.plot_tree(model,filled=True)
    plt.savefig(sys.argv[4]+'/1_c_bestTree')
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

file=open(sys.argv[4]+'/1_c.txt', 'w')

#setting fig wide as it is tall
fig = plt.figure(figsize=(12,12))

#cleaning train data
X_train,Y_train=cleanData(data=train_data)
X_test,Y_test=cleanData(data=test_data)
X_val,Y_val=cleanData(data=val_data)
train_data=None
test_data=None
val_data=None
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
model = tree.DecisionTreeClassifier()
path = model.cost_complexity_pruning_path(X_train, Y_train)

trees = []
nodeCount=[]
depth=[]
for alpha in path.ccp_alphas[:-1]:
    m=trainTree(X=X_train, Y=Y_train, alpha=alpha)
    nodeCount.append(m.tree_.node_count)
    depth.append(m.tree_.max_depth)
    trees.append(m)

# Plotting training data accuracy
train_accuracy=[]
for i in trees:
    train_accuracy.append(accuracy(X=X_train,Y=Y_train, model=i))

test_accuracy=[]
for i in trees:
    test_accuracy.append(accuracy(X=X_test,Y=Y_test, model=i))

val_accuracy=[]
for i in trees:
    val_accuracy.append(accuracy(X=X_val,Y=Y_val, model=i))

# Graph for total impurity vs effective alpha
ax1=fig.add_subplot(2,2,1)
ax1.plot(path.ccp_alphas[:-1], path.impurities[:-1], marker="x")
ax1.set_xlabel("effective alpha")
ax1.set_ylabel("total impurity of leaves")
ax1.set_title("total impurity vs effective alpha for training set")

# Graph for number of nodes vs alpha
ax2=fig.add_subplot(2,2,2)
ax2.plot(path.ccp_alphas[:-1], nodeCount, marker="x")
ax2.set_xlabel("alpha")
ax2.set_ylabel("number of nodes")
ax2.set_title("number of nodes vs alpha")

# Graph for number of nodes vs alpha
ax3=fig.add_subplot(2,2,3)
ax3.plot(path.ccp_alphas[:-1], depth, marker="x")
ax3.set_xlabel("alpha")
ax3.set_ylabel("depth of tree")
ax3.set_title("depth vs alpha")

# Graph for accuracy vs alpha
ax3=fig.add_subplot(2,2,4)
ax3.plot(path.ccp_alphas[:-1], train_accuracy, marker="x")
ax3.plot(path.ccp_alphas[:-1], test_accuracy, marker="o")
ax3.plot(path.ccp_alphas[:-1], val_accuracy, marker=".")
ax3.set_xlabel("alpha")
ax3.set_ylabel("accuracy")
ax3.set_title("accuracy vs alpha")
plt.savefig(sys.argv[4]+'/1_c_Graphs')
plt.tight_layout()
# plt.show()


# Best performaing model based on all three data splits
# maximum=0.0
# index=-1
# for i in range(len(train_accuracy)):
#     temp=train_accuracy[i]*test_accuracy[i]*val_accuracy[i]
#     if temp>maximum:
#         index=i
# print(path.ccp_alphas[index])

# Best performaing model based on validation data
index=val_accuracy.index(max(val_accuracy))
print("\nccp_alpha for best fitting model based on validation split: {}".format(path.ccp_alphas[index]))
file.write("ccp_alpha for best fitting model based on validation split: {}\n".format(path.ccp_alphas[index]))
# Tree for best performaing model
bestTree=trees[index]
print("Train accuracy of best tree: {}".format(train_accuracy[index]))
file.write("Train accuracy of best tree: {}\n".format(train_accuracy[index]))

print("Test accuracy of best tree: {}".format(test_accuracy[index]))
file.write("Test accuracy of best tree: {}\n".format(test_accuracy[index]))

print("Validation accuracy of best tree: {}".format(val_accuracy[index]))
file.write("Validation accuracy of best tree: {}\n".format(val_accuracy[index]))

plotTree(bestTree)
print('best tree is saved as bestTree.png')

file.close()

# dumping best ccp_alpha to use in further questions
with open(sys.argv[4]+'/1_c.best_ccp_alpha', 'wb') as f:
    pickle.dump(path.ccp_alphas[index],f)