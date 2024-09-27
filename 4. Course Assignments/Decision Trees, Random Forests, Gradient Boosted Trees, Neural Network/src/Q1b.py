import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
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

def trainTree(X,Y,**arg):
    # Training the model
    model=tree.DecisionTreeClassifier(**arg)
    model=model.fit(X,Y)
    return model

def plotTree(model):
    plt.figure(figsize=(12, 9))
    tree.plot_tree(model,filled=True)
    plt.savefig(sys.argv[4]+'/BestTree1_b')
    # plt.show()


# Reading Training data
with open(sys.argv[1],'r') as f:
    train_data=pd.read_csv(f)

# Reading Test data
with open(sys.argv[3],'r') as f:
    test_data=pd.read_csv(f)

# Reading Validation data
with open(sys.argv[2],'r') as f:
    val_data=pd.read_csv(f)

file=open(sys.argv[4]+'/1_b.txt', 'w')

#cleaning train data
X_train,Y_train=cleanData(data=train_data)
X_test,Y_test=cleanData(data=test_data)
X_val,Y_val=cleanData(data=val_data)
train_data=None
test_data=None
val_data=None

# Finding the best parameters using gridSearchCV
model=GridSearchCV(tree.DecisionTreeClassifier(),{
    'criterion':['gini', 'entropy', 'log_loss'],
    'max_depth': [i for i in range(2,11)],
    'min_samples_split': [i for i in range(2,11)],
    'min_samples_leaf': [i for i in range(1,11)],
    'splitter':['best','random'] },refit=True,n_jobs=-1,cv=3)
output=model.fit(X_train,Y_train)
# print('OPRIMAL PARAMS FOUND and the results of GridSearch have been saved to a csv file')
# results=pd.DataFrame(output.cv_results_)
# results=results[[
#     'param_criterion',
#     'param_max_depth',
#     'param_min_samples_split',
#     'param_min_samples_leaf',
#     'param_splitter',
#     'mean_test_score']]
# results.to_csv('GridSearchTrain.csv')

# Training best model
params=output.best_params_
print('Best Parameters\n{}'.format(params))
file.write('Best Parameters\n{}'.format(params))
bestModel=output.best_estimator_

# train data accuracy
acc=bestModel.score(X_train,Y_train)*100
print('Training data accuracy: {}'.format(acc))
file.write('Training data accuracy: {}\n'.format(acc))
# test data accuracy
acc=bestModel.score(X_test,Y_test)*100
print('Test data accuracy: {}'.format(acc))
file.write('Test data accuracy: {}\n'.format(acc))
# validation data accuracy
acc=bestModel.score(X_val,Y_val)*100
print('Validation data accuracy: {}'.format(acc))
file.write('Validation data accuracy: {}\n'.format(acc))
plotTree(bestModel)

file.close()
# dunping best params to use in further questions
with open(sys.argv[4]+'/1_b.bestParams', 'wb') as f:
    pickle.dump(params,f)