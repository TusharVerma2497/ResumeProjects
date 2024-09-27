import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import tracemalloc
import re
import operator
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from scipy import sparse
import sys

STOPWORDS.add('')
STOPWORDS.add('I')
STOPWORDS.add('it')

def trainTree(X,Y, alpha=0.0):
    # Training the model
    model=tree.DecisionTreeClassifier(ccp_alpha=alpha)
    model=model.fit(X,Y)
    return model
  
def accuracy(X,Y, model):
    # Y=Y.reshape(len(Y))
    pre=model.predict(X)
    correct=0
    for i in zip(pre,Y):
        # print(i)
        if(i[0]==i[1]):
            correct+=1
    return(correct*100/len(Y))

def cleanData(data=pd.DataFrame(), createVocab=False):
    global index_condition
    listToReplace=[i for i in range(len(index_condition))]
    data['condition'].replace(index_condition,listToReplace,inplace=True)
    data_index=[]
    for i in range(len(data['condition'])):
        if(data['condition'].iloc[i] not in listToReplace):
            data_index.append(i)
    data.drop(data_index,inplace=True)
    # del(index_condition)
    data.dropna()
    vocab=dict()
    cleanedReview=[]
    index=0

    for i in range(len(data['review'])):
        s=data['review'].iloc[i].lower().splitlines()
        #cleaning sentences of special characters
        s=list(map(lambda x: re.sub("[?;:,*/.()!\"<>@}#\\&%{-]",'',x),s))
        cleanedReview.append(s[0])

        if(len(cleanedReview)==20000):
            data['review'][index:i+1]=cleanedReview
            cleanedReview=[]
            index=i+1

        if(createVocab):
            words=set(s[0].split(' '))
            words=words-STOPWORDS

            for k in words:
                vocab.setdefault(k,0)
                vocab[k]+=1

    data['review'][index:i+2]=cleanedReview
    del cleanedReview

    if(createVocab):
        return(data,vocab)
    else:
        return data
# del(wordFreq)

def removeStopWords(data=pd.DataFrame(), STOPWORDS=set(), vocab=set(), TrainingData=True):
    stopWRemovedReview=[]
    index=0

    for i in range(len(data['review'])):
        words=set(data['review'].iloc[i].split(' '))
        words=words-STOPWORDS
        if(not TrainingData):
            words=words.intersection(vocab)
        stopWRemovedReview.append(' '.join(words))

        if(len(stopWRemovedReview)==20000):
            data['review'][index:i+1]=stopWRemovedReview
            stopWRemovedReview=[]
            index=i+1

    data['review'][index:i+2]=stopWRemovedReview
    del stopWRemovedReview
    return data

def vectorizer(data=pd.DataFrame(), TrainingData=True,vocab=None):
    #initializing the count Vectorizer
    cv=CountVectorizer(vocabulary=vocab)
    X_train=cv.fit_transform(data['review'])
    if(TrainingData):
        vocab=cv.vocabulary_
    X_train=sparse.lil_matrix(X_train)
    j=X_train.shape[1]-1
    for i in range(len(data['review'])):
        X_train[(i,j)]=data['condition'].iloc[i]
    Y_train=data['rating']
    if(TrainingData):
        return (X_train,Y_train,vocab)
    else:
        return (X_train,Y_train)

def testModel(test_data=pd.DataFrame(), STOPWORDS=set(), vocab=dict()):
    test_data=cleanData(data=test_data)
    test_data=removeStopWords(data=test_data, STOPWORDS=STOPWORDS,vocab=set(vocab.keys()), TrainingData=False)
    X_test, Y_test=vectorizer(data=test_data, TrainingData=False, vocab=vocab)
    return (X_test,Y_test)

file=open(sys.argv[4]+'/2_c.txt', 'w')
# Reading Training data
with open(sys.argv[1],'r') as f:
    train_data=pd.read_csv(f)

index_condition=train_data['condition'].unique()
# cleaning data and creating vocab from training data
train_data, vocab=cleanData(data=train_data, createVocab=True)
# extracting top 10000 vobac as features for our classifier
vocab=dict(sorted(vocab.items(), key=operator.itemgetter(1), reverse=True))
# At the same time adding all the other words to the STOPWORDS
for i in list(vocab.keys())[250:]:
    STOPWORDS.add(i)
vocab=list(vocab.keys())[0:250]
# vocab=list(vocab.keys())

# Removing all the STOPWORDS from training data 
train_data=removeStopWords(data=train_data,STOPWORDS=STOPWORDS)
X_train, Y_train, vocab=vectorizer(data=train_data)
del train_data
print('Performing post complexity pruning....')
model = tree.DecisionTreeClassifier()
path = model.cost_complexity_pruning_path(X_train, Y_train)
# modelTrain=trainTree(X=X_train,Y=Y_train)

cutDown_ccp_alpha=list(path.ccp_alphas[::1300])
cutDown_ccp_alpha.append(path.ccp_alphas[-1100])
cutDown_ccp_alpha.append(path.ccp_alphas[-900])
cutDown_ccp_alpha.append(path.ccp_alphas[-500])
cutDown_ccp_alpha.append(path.ccp_alphas[-250])
cutDown_ccp_alpha.append(path.ccp_alphas[-100])
cutDown_ccp_alpha.append(path.ccp_alphas[-2])

print('Training {} trees'.format(len(cutDown_ccp_alpha)))

trees = []
nodeCount=[]
depth=[]
for alpha in cutDown_ccp_alpha:
    print('Training model: {} for alpha value: {}'.format(len(trees)+1,alpha))
    m=trainTree(X=X_train, Y=Y_train, alpha=alpha)
    nodeCount.append(m.tree_.node_count)
    depth.append(m.tree_.max_depth)
    trees.append(m)
    print('Training done | nodes: {} depth:{}'.format(m.tree_.node_count,m.tree_.max_depth))

# #Saving the model for convinience
# with open('path','wb') as f:
#     pickle.dump((path,model,trees,nodeCount,depth),f)

# with open('path','rb') as f:
#     path,model,trees,nodeCount,depth=pickle.load(f)

#setting fig wide as it is tall
fig = plt.figure(figsize=(12,12))

# Graph for total impurity vs effective alpha
ax1=fig.add_subplot(2,2,1)
ax1.plot(path.ccp_alphas[:-1], path.impurities[:-1], marker="x")
ax1.set_xlabel("effective alpha")
ax1.set_ylabel("total impurity of leaves")
ax1.set_title("total impurity vs effective alpha for training set")

# Graph for number of nodes vs alpha
ax2=fig.add_subplot(2,2,2)
ax2.plot(cutDown_ccp_alpha, nodeCount, marker="x")
ax2.set_xlabel("alpha")
ax2.set_ylabel("number of nodes")
ax2.set_title("number of nodes vs alpha")

# Graph for number of nodes vs alpha
ax3=fig.add_subplot(2,2,3)
ax3.plot(cutDown_ccp_alpha, depth, marker="x")
ax3.set_xlabel("alpha")
ax3.set_ylabel("depth of tree")
ax3.set_title("depth vs alpha")


# Plotting training data accuracy
train_accuracy=[]
for i in trees:
    train_accuracy.append(accuracy(X=X_train,Y=Y_train, model=i))
del X_train
del Y_train

# Test data accuracy
with open(sys.argv[3],'r') as f:
    test_data=pd.read_csv(f)
X_test, Y_test=testModel(test_data=test_data, STOPWORDS=STOPWORDS, vocab=vocab)
test_accuracy=[]
for i in trees:
    test_accuracy.append(accuracy(X=X_test,Y=Y_test, model=i))
del X_test
del Y_test

# Validation data accuracy
with open(sys.argv[2],'r') as f:
    test_data=pd.read_csv(f)
X_val, Y_val=testModel(test_data=test_data, STOPWORDS=STOPWORDS, vocab=vocab)
val_accuracy=[]
for i in trees:
    val_accuracy.append(accuracy(X=X_val,Y=Y_val, model=i))
del X_val
del Y_val
del test_data

# Graph for accuracy vs alpha
ax3=fig.add_subplot(2,2,4)
ax3.plot(cutDown_ccp_alpha, train_accuracy, marker="x")
ax3.plot(cutDown_ccp_alpha, test_accuracy, marker="o")
ax3.plot(cutDown_ccp_alpha, val_accuracy, marker=".")
ax3.set_xlabel("alpha")
ax3.set_ylabel("accuracy")
ax3.set_title("accuracy vs alpha")
plt.savefig('2_c_CCP_AlphaGraphs')
plt.tight_layout()
print('All the asked plots are saved as Graphs.png')
# plt.show()

# Best performaing model based on validation data
index=val_accuracy.index(max(val_accuracy))
print("ccp_alpha for best fitting model based on validation split: {}".format(path.ccp_alphas[index]))
file.write("ccp_alpha for best fitting model based on validation split: {}\n".format(path.ccp_alphas[index]))
# Tree for best performaing model
bestTree=trees[index]
print("Train accuracy of best tree: {}".format(train_accuracy[index]))
file.write("Train accuracy of best tree: {}\n".format(train_accuracy[index]))

print("Test accuracy of best tree: {}".format(test_accuracy[index]))
file.write("Test accuracy of best tree: {}\n".format(test_accuracy[index]))

print("Validation accuracy of best tree: {}".format(val_accuracy[index]))
file.write("Validation accuracy of best tree: {}\n".format(val_accuracy[index]))

file.close()
# ccp_alpha for best fitting model based on validation split: 0.0
# Train accuracy of best tree: 99.39154001487937
# Test accuracy of best tree: 54.92391081639875
# Validation accuracy of best tree: 55.21402550091074