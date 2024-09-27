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
    # Training the model
    print('Training')
    # model=tree.DecisionTreeClassifier()
    # model=model.fit(X,Y)
    model=GridSearchCV(RandomForestClassifier(bootstrap=True,oob_score=True),{
    'n_estimators':[i for i in range(50,451,50)],
    # 'n_estimators':[450],
    'max_features': [0.4, 0.5, 0.6, 0.7, 0.8],
    # 'max_features': [0.4, 0.8],
    'min_samples_split': [i for i in range(2,11,2)] },
    # 'min_samples_split': [6,10] },
    refit=True, verbose=10, n_jobs=-1, cv=2)
    output=model.fit(X,Y)
    print('Training done\nBest parameters:\n{}'.format(output.best_params_))
    # Returing the best model
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


tracemalloc.start()
file=open(sys.argv[4]+'/2_d.txt', 'w')
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

# #Saving the model for convinience
# with open('model300','wb') as f:
#     pickle.dump((modelTrain,vocab,index_condition,STOPWORDS),f)

# with open('model300','rb') as f:
#     modelTrain,vocab,index_condition,STOPWORDS=pickle.load(f)



# Testing Data
# OOB_score
print('OOB Score: {}'.format(modelTrain.oob_score_*100))
file.write('OOB Score: {}\n'.format(modelTrain.oob_score_*100))

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

# {'max_features': 0.4, 'min_samples_split': 6, 'n_estimators': 50}
# OOB Score: 59.59807985262338
# Training data accuracy: 98.3694689481702
# Test data accuracy: 61.31838247620467
# Validation data accuracy: 61.100761715515816