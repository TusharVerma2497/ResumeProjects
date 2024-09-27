import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import sys

def cleanData(data):
    # dropping the missing value
    data.replace({'?': np.nan},inplace =True)
    data=data.dropna(axis=0,how='any')

    # extracting input and output
    Y=np.array(data[['Severity']])
    Y=Y.reshape(len(Y))
    X=data.drop(['BI-RADS assessment','Severity'],axis=1)
    return (X,Y)

# def trainTree(X,Y,alpha):
#     # Training the model
#     # print('Training start')
#     model=tree.DecisionTreeClassifier(ccp_alpha=alpha)
#     model=model.fit(X,Y)
#     return model

# def plotTree(model):
#     plt.figure(figsize=(12, 9))
#     tree.plot_tree(model,filled=True)
#     plt.savefig('bestTree')
#     # plt.show()
    
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

file=open(sys.argv[4]+'/1_d.txt', 'w')


#cleaning train data
X_train,Y_train=cleanData(data=train_data)
X_test,Y_test=cleanData(data=test_data)
X_val,Y_val=cleanData(data=val_data)
train_data=None
test_data=None
val_data=None

# Finding the best parameters using gridSearchCV
model=GridSearchCV(RandomForestClassifier(bootstrap=True,oob_score=True),{
    'criterion':['gini', 'entropy', 'log_loss'],
    'n_estimators':[i for i in range(30,100,20)],
    'max_features': [2,3,4,'sqrt', 'log2'],
    'min_samples_split': [i for i in range(2,6)],
    'min_samples_leaf': [i for i in range(1,6)] },refit=True,cv=2,n_jobs=-1)
output=model.fit(X_train,Y_train)

# print('OPRIMAL PARAMS FOUND and the results of GridSearch have been saved to a csv file')
# results=pd.DataFrame(output.cv_results_)
# results=results[[
#     'param_criterion',
#     'param_n_estimators',
#     'param_max_features',
#     'param_min_samples_split',
#     'param_min_samples_leaf',
#     'mean_test_score']]
# results.to_csv('GridSearchTrain1.csv')


# # saving the model
# with open('model', 'wb') as f:
#     pickle.dump(output,f)

# with open('model', 'rb') as f:
#     output=pickle.load(f)

params=output.best_params_
print("Best parameters\n{}".format(params))
file.write("Best parameters{}\n".format(params))

bestModel=output.best_estimator_
print('OOB Score: {}'.format(bestModel.oob_score_*100))
file.write('OOB Score: {}\n'.format(bestModel.oob_score_*100))

# train data accuracy
acc=output.score(X_train,Y_train)*100
print('Training data accuracy: {}'.format(acc))
file.write('Training data accuracy: {}\n'.format(acc))

# test data accuracy
acc=output.score(X_test,Y_test)*100
print('Test data accuracy: {}'.format(acc))
file.write('Test data accuracy: {}\n'.format(acc))

# validation data accuracy
acc=output.score(X_val,Y_val)*100
print('Validation data accuracy: {}'.format(acc))
file.write('Validation data accuracy: {}\n'.format(acc))

file.close()
# dunping best params to use in further questions
with open(sys.argv[4]+'/1_d.bestParams', 'wb') as f:
    pickle.dump(params,f)