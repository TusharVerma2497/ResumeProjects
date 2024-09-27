import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import tree
import statistics
import sys

def myMode(l):
    return int(statistics.mode(l))
def cleanData(data,func):
    # dropping the missing value
    data.replace({'?': np.nan},inplace =True)
    # data=data.dropna(axis=0,how='any')
    myMode(data['Density'])
    # exit(0)
    data=data.fillna({
        'Age': round(func(data['Age'])),
        'Shape': round(func(data['Shape'])),
        'Margin': round(func(data['Margin'])),
        'Density': round(func(data['Density'])),
        'Severity': round(func(data['Severity']))
    })
    # extracting input and output
    Y=np.array(data[['Severity']])
    X=data.drop(['BI-RADS assessment','Severity'],axis=1)
    return (X,Y)

def trainTree(X,Y,alpha=0.0):
    # Training the model
    model=tree.DecisionTreeClassifier(ccp_alpha=alpha)
    model=model.fit(X,Y)
    return model

def plotTree(model,name=''):
    plt.figure(figsize=(12, 9))
    tree.plot_tree(model,filled=True)
    plt.savefig(sys.argv[4]+"/"+name+'bestTree')
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

file=open(sys.argv[4]+'/1_e.txt', 'w')

#cleaning train data with median
X_train_med,Y_train_med=cleanData(data=train_data,func=pd.DataFrame.median)
X_test_med,Y_test_med=cleanData(data=test_data,func=pd.DataFrame.median)
X_val_med,Y_val_med=cleanData(data=val_data,func=pd.DataFrame.median)
#cleaning train data with mode
X_train_mod,Y_train_mod=cleanData(data=train_data,func=myMode)
X_test_mod,Y_test_mod=cleanData(data=test_data,func=myMode)
X_val_mod,Y_val_mod=cleanData(data=val_data,func=myMode)
train_data=None
test_data=None
val_data=None

def probA(X_train, Y_train, X_test, Y_test, X_val, Y_val, funcName='Median'):
    global file
    mode='median' if 'Median' in funcName else 'mode'
    # Training the model
    modelTrain=trainTree(X=X_train,Y=Y_train)

    #plotting the tree
    plotTree(modelTrain,name=funcName)

    #Train data accuracy
    print("\n Q1a: na valuse are filled with "+mode)
    file.write("\n Q1a: na valuse are filled with "+mode+" values\n")

    acc=accuracy(X_train,Y_train, modelTrain)
    print("Accuracy of Training data: {}".format(acc))
    file.write("Accuracy of Training data: {}\n".format(acc))

    #Test data accuracy
    acc=accuracy(X_test,Y_test, modelTrain)
    print("Accuracy of Test data: {}".format(acc))
    file.write("Accuracy of Test data: {}\n".format(acc))

    #Validation data accuracy
    acc=accuracy(X_val,Y_val, modelTrain)
    print("Accuracy of Validation data: {}".format(acc))
    file.write("Accuracy of Validation data: {}\n".format(acc))
    file.flush()

# Q1a replacing na with median 
probA(X_train_med,Y_train_med,X_test_med,Y_test_med,X_val_med,Y_val_med, funcName='1_e_Q1aMedian_')
# Q1a replacing na with mode
probA(X_train_mod,Y_train_mod,X_test_mod,Y_test_mod,X_val_mod,Y_val_mod, funcName='1_e_Q1aMode_')



def probB(X_train, Y_train, X_test, Y_test, X_val, Y_val, funcName='Median'):
    global file
    mode='median' if 'Median' in funcName else 'mode'

    print("\n Q1b: na valuse are filled with "+mode)
    file.write("\n Q1b: na valuse are filled with "+mode+" values\n")
    # Finding the best parameters using gridSearchCV
    model=GridSearchCV(tree.DecisionTreeClassifier(),{
        'criterion':['gini', 'entropy', 'log_loss'],
        'max_depth': [i for i in range(2,11)],
        'min_samples_split': [i for i in range(2,11)],
        'min_samples_leaf': [i for i in range(1,11)],
        'splitter':['best','random'] },refit=True, cv=2, n_jobs=-1)
    output=model.fit(X_train,Y_train)
    # Training best model
    params=output.best_params_
    bestModel=output.best_estimator_

    params=output.best_params_
    print('Best Parameters\n{}'.format(params))
    file.write('Best Parameters{}\n'.format(params))

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
    plotTree(bestModel,name=funcName)
    file.flush()

# Q1b replacing na with median 
probB(X_train_med,Y_train_med,X_test_med,Y_test_med,X_val_med,Y_val_med, funcName='1_e_Q1bMedian_')
# Q1b replacing na with mode
probB(X_train_mod,Y_train_mod,X_test_mod,Y_test_mod,X_val_mod,Y_val_mod, funcName='1_e_Q1bMode_')



def probC(X_train, Y_train, X_test, Y_test, X_val, Y_val, funcName='Median'):
    global file
    mode='median' if 'Median' in funcName else 'mode'
    print("\n Q1c: na valuse are filled with "+mode)
    file.write("\n Q1c: na valuse are filled with "+mode+" values\n")

    fig = plt.figure(figsize=(12,12))
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
    plt.savefig(sys.argv[4]+'/'+"1_c_"+mode+'Graphs')
    plt.tight_layout()
    print('All the asked plots are saved as Graphs.png')
    # plt.show()

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
    plotTree(bestTree,funcName)
    file.flush()

# Q1c replacing na with median 
probC(X_train_med,Y_train_med,X_test_med,Y_test_med,X_val_med,Y_val_med, funcName='1_e_Q1cMedian_')
# Q1c replacing na with mode
probC(X_train_mod,Y_train_mod,X_test_mod,Y_test_mod,X_val_mod,Y_val_mod, funcName='1_e_Q1cMode_')



def probD(X_train, Y_train, X_test, Y_test, X_val, Y_val, funcName='Median'):
    global file
    mode='median' if 'Median' in funcName else 'mode'
    print("\n Q1d: na valuse are filled with "+mode)
    file.write("\n Q1d: na valuse are filled with "+mode+" values\n")
    # Finding the best parameters using gridSearchCV
    model=GridSearchCV(RandomForestClassifier(bootstrap=True,oob_score=True),{
        'criterion':['gini', 'entropy', 'log_loss'],
        'n_estimators':[i for i in range(30,100,20)],
        'max_features': [2,3,4,'sqrt', 'log2'],
        'min_samples_split': [i for i in range(2,6)],
        'min_samples_leaf': [i for i in range(1,6)] },refit=True,cv=2,n_jobs=-1)
    output=model.fit(X_train,np.ravel(Y_train))

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

    # Training best model
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
    file.flush()

# Q1d replacing na with median 
probD(X_train_med,Y_train_med,X_test_med,Y_test_med,X_val_med,Y_val_med, funcName='1_e_Q1dMedian_')
# Q1d replacing na with mode
probD(X_train_mod,Y_train_mod,X_test_mod,Y_test_mod,X_val_mod,Y_val_mod, funcName='1_e_Q1dMode_')

file.close()