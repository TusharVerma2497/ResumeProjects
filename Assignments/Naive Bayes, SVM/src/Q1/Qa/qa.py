import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import os
import re
import pickle
import math 
import random
import nltk
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
import sys

#creating stemming object
ps = PorterStemmer()

def trainParameters(loc):
    vocab=dict()
    l=os.listdir(loc)
    c=0
    totalDocuments=len(l)
    for i in l:
        # words=set()
        with open(loc+i) as file:
            sentences=file.read().splitlines()
            #cleaning sentences of special characters
            sentences=list(map(lambda x: re.sub("[?;:,*/.()!\"<>@}#\\&%{-]",'',x),sentences))
            words=sentenceTreatment(sentences=sentences,stopwords=False,stemming=False,bigram=False)
            for k in words:
                vocab.setdefault(k,0)
                vocab[k]+=1
        c+=1
       #  print(c)
    # # taking count of top N most frequent words only
    # wordFreq=dict(sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)[:N])
    wordFreq=dict(sorted(vocab.items(), key=operator.itemgetter(1), reverse=True))
    #finding probability of each word/document and applying 
    #the Laplace smoothing at the same time 
    #here k=2 because we have 2 sentiments positive and negative
    wordFreq={key:(value+1)/(totalDocuments+2) for key, value in wordFreq.items()}
    return wordFreq


def sentenceTreatment(sentences=[], stopwords=False, stemming=False, bigram=False):
    words=set()
    for j in sentences:
        temp=set(j.split(' '))
        # performing stemming on the words
        if(stemming):
            temp=set(map(lambda x: ps.stem(x),temp))
        # removing STOPWORDS
        if(stopwords):
            temp=temp-STOPWORDS
        # performing Bi-gram
        if(bigram):
            temp1=nltk.bigrams(temp)
            bigrams=set()
            for k in temp1:
                bigrams.add(' '.join(k))
            words=words.union(bigrams)
        words=words.union(temp)
    return words


def predict(file,models=[],classProb=[]):

    sentences=file.read().splitlines()
    #cleaning sentences of special characters
    sentences=list(map(lambda x: re.sub("[?;:,*/.()!\"<>-]",'',x),sentences))
    words=sentenceTreatment(sentences=sentences,stopwords=False,stemming=False,bigram=False)
    #calculating the log likelihood for each model
    logLikelihood=[] 
    for i in range(len(models)):
        sum=0
        #default probabilty of the words which doesen't appear while training 
        #applying laplace smoothing
        default=1.0/12502.0
        for j in words:
            sum+=math.log(models[i].setdefault(j,default))
        logLikelihood.append(sum+math.log(classProb[i]))

    return logLikelihood


 
def test(loc,models=[]):
    #Initializing the confusion matrix 
    confusionMatrix=np.zeros(len(models)**2).reshape(len(models),len(models))
    totalDocuments=0.0
    for i in loc:
        l=os.listdir(i)
        totalDocuments+=len(l)
        for j in l:
            with open(i+j) as file:
                expectedOutput= 1 if "pos" in i else 0
                # uncomment this for random guessing the class of the document
                # expectedOutput=random.randint(0,1)
                # uncomment this for assigning 1 as the class of the document
                # expectedOutput=1
                t=predict(file,models=models,classProb=[0.5,0.5])
                # if predicted class is negative
                if t[0]>t[1]:
                    #True Negative
                    if(expectedOutput==0):
                        confusionMatrix[1][1]+=1
                    #False Positive
                    else:
                        confusionMatrix[0][1]+=1
                #if predicted class is positive
                else:
                    #True Positive
                    if(expectedOutput==1):
                        confusionMatrix[0][0]+=1
                    #False Negative
                    else:
                        confusionMatrix[1][0]+=1
    return confusionMatrix


def displayWordCloud(model,title):
# creating string for wordcloud
    s=''
    c=0
    for i,j in model.items():
        i=i+' '
        s+=i*int(j*12502.0)
        c+=1
        if(c>20000):
            break
    s=s.split(' ')
    random.shuffle(s)
    s=' '.join(word for word in s)

# initializing wordcloud object
    wordcloud = WordCloud(width = 1200, height = 1000,
                background_color ='white',
                stopwords = set(), #we already removed stopwords earlier
                min_font_size = 2).generate(s)

#plotting the graph using matplotlib
    plt.figure(figsize = (12, 10), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(title)
    print("saved image "+title)
    # plt.show()


# #Training the model and returning top N=20000 most frequent occuring words in a class
trainDataDir=sys.argv[1]
testDataDir=sys.argv[2]

#reading independent variables from X.csv and dependent variable from Y.csv and normalizing them
positiveTrainPara=trainParameters(loc=os.path.join(trainDataDir,'pos/'))
negativeTrainPara=trainParameters(loc=os.path.join(trainDataDir,'neg/'))

# Store the model so that we can use the trained model
with open('positiveModel','wb') as file:
    pickle.dump(positiveTrainPara,file)

with open('negativeModel','wb') as file:
    pickle.dump(negativeTrainPara,file)



#Testing data

# using deserialization to load the model
with open('positiveModel','rb') as file:
    positiveTrainPara=pickle.load(file)

with open('negativeModel','rb') as file:
    negativeTrainPara=pickle.load(file)
print("Done-Training")


#accuracy of training data
res=test(loc=[os.path.join(trainDataDir,'pos/'),os.path.join(trainDataDir,'neg/')],models=[negativeTrainPara, positiveTrainPara])
accuracy=(res[0][0]+res[1][1])*100/(res[0][0]+res[1][1]+res[0][1]+res[1][0])
print("Training Data Accuracy: {}".format(accuracy))
# accuracy of test data
res=test(loc=[os.path.join(testDataDir,'pos/'),os.path.join(testDataDir,'neg/')],models=[negativeTrainPara, positiveTrainPara])
accuracy=(res[0][0]+res[1][1])*100/(res[0][0]+res[1][1]+res[0][1]+res[1][0])
print("Test Data Accuracy: {}".format(accuracy))

# Constructing the word cloud
displayWordCloud(model=negativeTrainPara,title='Negative Model')
displayWordCloud(model=positiveTrainPara,title='Positive Model')
