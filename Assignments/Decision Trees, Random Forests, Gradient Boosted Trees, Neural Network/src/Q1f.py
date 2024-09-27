import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import xgboost
import sys

def cleanData(data):
    # dropping the missing value
    data.replace({'?': np.nan},inplace =True)
    # data=data.dropna(axis=0,how='any')

    # extracting input and output
    Y=np.array(data[['Severity']],dtype='float64')
    Y=Y.reshape(len(Y))
    X=data.drop(['BI-RADS assessment','Severity'],axis=1)
    return (X,Y)


# Reading Training data
with open(sys.argv[1],'r') as f:
    train_data=pd.read_csv(f)

# Reading Test data
with open(sys.argv[3],'r') as f:
    test_data=pd.read_csv(f)

# Reading Validation data
with open(sys.argv[2],'r') as f:
    val_data=pd.read_csv(f)

file=open(sys.argv[4]+'/1_f.txt', 'w')

def changeDtype(df):
    for i in df.columns:
        df[i]=df[i].astype(float)
    return df

#cleaning train data
X_train,Y_train=cleanData(data=train_data)
X_test,Y_test=cleanData(data=test_data)
X_val,Y_val=cleanData(data=val_data)
train_data=None
test_data=None
val_data=None

X_train=changeDtype(X_train)
X_test=changeDtype(X_test)
X_val=changeDtype(X_val)


# Finding the best parameters using gridSearchCV
model=GridSearchCV(xgboost.XGBClassifier(),{
    'n_estimators':[i for i in range(10,51,10)],
    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'max_depth': [i for i in range(4,11,1)] },refit=True,n_jobs=-1,cv=2)
output=model.fit(X_train,Y_train)

# print('OPRIMAL PARAMS FOUND and the results of GridSearch have been saved to a csv file')
# results=pd.DataFrame(output.cv_results_)
# results=results[[
#     'param_n_estimators',
#     'param_subsample',
#     'param_max_depth',
#     'mean_test_score']]
# results.to_csv('GridSearchTrain.csv')


# # saving the model
# with open('model', 'wb') as f:
#     pickle.dump(output,f)

# with open('model', 'rb') as f:
#     output=pickle.load(f)

params=output.best_params_
print("Best parameters\n{}".format(params))
file.write("Best parameters{}\n".format(params))

bestModel=output.best_estimator_

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