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

def trainTree(X,Y):
    global file
    # Training the model
    print('Training')
    model=GridSearchCV(tree.DecisionTreeClassifier(),{
    # 'criterion':['gini', 'entropy', 'log_loss'],
    'max_depth': [i for i in range(90,101,5)],
    'min_samples_split': [1,2,4,6,8,10],
    'min_samples_leaf': [1,3,5,7,9],
    # 'splitter':['best','random'],
     },refit=True,verbose=10, n_jobs=-1, cv=2)
    output=model.fit(X,Y)
    print('Training done')
    params=output.best_params_
    print('Best Parameters\n{}'.format(params))
    file.write('Best Parameters: {}\n'.format(params))
    # Return best model
    return output.best_estimator_
  
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
    res=accuracy(X_test, Y_test, modelTrain)
    return res


file=open(sys.argv[4]+'/2_b.txt', 'w')
tracemalloc.start()
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
modelTrain=trainTree(X=X_train,Y=Y_train)
del X_train
del Y_train

#Saving the model for convinience
# with open('model1000','wb') as f:
#     pickle.dump((modelTrain,vocab,index_condition,STOPWORDS),f)

# with open('model1000','rb') as f:
#     modelTrain,vocab,index_condition,STOPWORDS=pickle.load(f)



# Testing Data
# Training data accuracy
with open(sys.argv[1],'r') as f:
    test_data=pd.read_csv(f)
res=testModel(test_data=test_data, STOPWORDS=STOPWORDS, vocab=vocab)
print("Training data accuracy: "+str(res))
file.write("Training data accuracy: "+str(res)+"\n")

# Test data accuracy
with open(sys.argv[3],'r') as f:
    test_data=pd.read_csv(f)
res=testModel(test_data=test_data, STOPWORDS=STOPWORDS, vocab=vocab)
print("Test data accuracy: "+str(res))
file.write("Test data accuracy: "+str(res)+"\n")

# Validation data accuracy
with open(sys.argv[2],'r') as f:
    test_data=pd.read_csv(f)
res=testModel(test_data=test_data, STOPWORDS=STOPWORDS, vocab=vocab)
print("Validation data accuracy: "+str(res))
file.write("Validation data accuracy: "+str(res)+"\n")


print(tracemalloc.get_traced_memory())
tracemalloc.stop()

file.close()

# {'max_depth': 100, 'min_samples_leaf': 1, 'min_samples_split': 2}
# Training data accuracy: 99.03726928118468
# Test data accuracy: 54.849404883863876
# Validation data accuracy: 55.069133962576586