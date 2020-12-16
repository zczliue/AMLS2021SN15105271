


# This is the function file for Task B1,
# which assess the performance between different classifiers, 
# by grid search with cross validation, including:

# Logistic Regression
# SVM 
# Random Forest
# KNN
# MLP

# The assessment metrices for classifiers include:
# 1. Accuracy score between ytrue and ypred
# 2. Grid score for grid search tuning
# 


# To use this file, specify:
# 'image_folder' : a string in either 'celeba' or 'cartoon_set'
# 'label_name' :  a string that specifies the feature name, i.e. 'gender', 'smiling' as in labels.csv
# 'tuning_parameters' : a dictionary contains different kernels and settings
# 'cvfold': an int for K-fold cross validation

# return value: trained classifier with best parameter settings



import os
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import keras
import sklearn
import cv2
import dlib
#import rewritten lab2 landmarks extraction 
from B1 import B1_extract_landmarks as B1_extract
import itertools 
import datetime
import pickle



from itertools import cycle
from numpy import interp

from sklearn import datasets, metrics, model_selection, svm
from sklearn.utils import shuffle
from sklearn.datasets import load_iris

from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import roc_curve, auc

from keras.preprocessing import image
from keras.models import Sequential 
from keras.models import Sequential 
from keras.layers import Dense , Dropout 
from keras.optimizers import SGD
from keras.optimizers import RMSprop



# Set the parameters for Grid search
split_ratio = 0.75
cvfold = 5
image_folder = 'cartoon_set'
label_name   = 'face_shape'
num_epochs = 40


# we do not need Log reg in B1
#LogReg_tuned_parameters = {'solver':[ 'saga'], 'penalty':['l1', 'l2'], 'C':[1e-4, 1e-3, 1e-2, 1e-1, 1,10]}  

SVM_tuned_parameters = {'kernel': ['rbf', 'linear'],
                        'C': [0.001, 0.01, 0.1, 1, 10],
                        'gamma': [1e-4, 1e-3, 1e-2, 1e-1]}
                 

#Random Forest
RF_tuned_parameters = [{'n_estimators':[64, 128, 256, 512, 1024], 'max_depth':[4, 8, 16, 32, 64]}]

#KNN parameter
KNN_tuned_parameters = [{'n_neighbors':[8,16,32,64,128]}]

#MLP parameter
MLP_tuned_parameters = [{'hidden_layer_sizes': [(128,256,128,), (128,256,128,), (128,),(256,),(512,)], 
                         'activation': ['logistic', 'tanh', 'relu'],
                         'learning_rate':['constant', 'adaptive']}]


Squential_model_parameters = [{'num_hidden_layer': 3,
                              'hidden_layer_activation' : ['relu','tanh','relu'],
                              'dropout':[0.5,0.25,0.125],
                              'last_activation':'softmax'},
                              
                              {'num_hidden_layer': 4,
                              'hidden_layer_activation' : ['relu','tanh','relu','tanh'],
                              'dropout':[0.5,0.25,0.125,0.0625],
                              'last_activation':'softmax'}]


#------------------------------------------Task B1 function definition-----------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------


# This function is rewritten as follows:
# the input image_folder must be a string that specifies 'celeba' or 'cartoon_set'
# the input label_name must be a string that specifies the feature name, i.e. 'gender', 'smiling' as in labels.csv

def split_data(X, y, split_ratio):
    
    X, Y = shuffle(X,y)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=split_ratio)
    
    #reshape into appropriate dimensions
    tr_X = train_X.reshape((train_X.shape[0], train_X.shape[1]*2))
    te_X = test_X.reshape((test_X.shape[0], test_X.shape[1]*2))
    
    
    #normalisation
    x_scaler = StandardScaler()
    tr_X = x_scaler.fit_transform(tr_X)
    te_X = x_scaler.transform(te_X)
    
    #train_Y, test_Y are single valued Y (multi-class) 
    tr_Y = train_Y
    te_Y = test_Y

    
    #convert to category matrix. 
    cat_tr_Y = keras.utils.to_categorical(tr_Y)
    cat_te_Y = keras.utils.to_categorical(te_Y)

    return tr_X, te_X, tr_Y, te_Y, cat_tr_Y, cat_te_Y 




def bulk_runtime_estimation(classifier, xtest, clftype):
    
    nsamp = xtest.shape[0]
    start = time.time()

    #Sequential.predict_proba (from tensorflow.python.keras.engine.sequential) 
    # is deprecated and will be removed after 2021-01-01.

    #for models other than Seq
    if clftype == 'other':
        #hard decision
        ypred = classifier.predict(xtest)
        #probability decision
        ypred_prob = classifier.predict_proba(xtest)
    else:
        #hard decision
        #ypred = classifier.predict_classes(xtest)
        ypred = np.argmax(classifier.predict(xtest), axis=-1)
        #probability
        ypred_prob = classifier.predict(xtest)


    #print(ypred)
    #print(ypred_prob)
    
    bulk_runtime = time.time() - start
    
    #average runtime per instance
    avg_runtime = bulk_runtime/nsamp
    
    return ypred, ypred_prob, bulk_runtime, avg_runtime
    



 

#Prediction with Grid search SVC Cross validation
def SVC_GridSearch_PredictionCV(xtrain, ytrain, xtest, ytest, tuning_parameters, cvfold, num_class, savepath):
    # classifier
    clf = GridSearchCV(SVC(probability=True), tuning_parameters, cv = cvfold, return_train_score=True)
    clf.fit(xtrain, ytrain)
    
    means_train = clf.cv_results_['mean_train_score']
    stds_train  = clf.cv_results_['std_train_score']    
        
    means_val = clf.cv_results_['mean_test_score']
    stds_val = clf.cv_results_['std_test_score'] 
    

    #Prediction on a pseudo test set (split from Dataset A) using the best estimator
    ypred, ypred_prob, bulk_runtime, avg_runtime = bulk_runtime_estimation(clf.best_estimator_, xtest, 'other')
    ytrue, ypred = ytest, ypred

    scores_on_best_para = {}
    print()
    print('-------------------------------------------------')
    print("SVM Grid search CV on Dataset A:")
    print()
    print("Training scores:")
    print()

    #print training scores
    for mean, std, params in zip(means_train, stds_train, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
        if params == clf.best_params_:
            scores_on_best_para['training'] = [mean, std]
        
    print()
    
    #print validation scores
    print("Validation scores:")
    print()
    #print grid search validation table
    for mean, std, params in zip(means_val, stds_val, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
        if params == clf.best_params_:
            scores_on_best_para['validation'] = [mean, std]

    print()
    

    print("Prediction on a pseudo test set (split from Dataset A):")
    print(classification_report(ytrue, ypred))
    pseudo_test_score = accuracy_score(ytrue, ypred)
    scores_on_best_para['pseudo_test'] = pseudo_test_score
    print("Accuracy:", pseudo_test_score)
    print()
    print("Best parameters found on Dataset A:")
    print()
    print(clf.best_params_)
    print()
    print('Average runtime per test instance:', avg_runtime)
    print('-------------------------------------------------')
    print()
    print()
    
    cat_ytrue = keras.utils.to_categorical(ytest)
    # confusion matrix
    plot_confusion_matrix(cat_ytrue, ypred_prob, 'SVM', savepath)
    # ROC curve
    plot_ROC_curve(cat_ytrue, ypred_prob, num_class, 'SVM', savepath)

    #plot learning curve for best para combination
    lctrain_sizes, lctrain_scores, lcvalid_scores = learning_curve(clf.best_estimator_, xtrain, ytrain, train_sizes=np.linspace(0.1, 1.0, 5), cv=cvfold)
    plot_learning_curves_other(lctrain_sizes, lctrain_scores,lcvalid_scores, 'SVM', savepath)
    
    #return best classifier
    return clf.best_estimator_, scores_on_best_para
    
    





def RF_GridSearch_PredictionCV(xtrain, ytrain, xtest, ytest, tuning_parameters, cvfold, num_class, savepath):

    clf = GridSearchCV(RandomForestClassifier() , tuning_parameters, cv = cvfold, return_train_score=True)
    clf.fit(xtrain, ytrain)
    
    means_train = clf.cv_results_['mean_train_score']
    stds_train  = clf.cv_results_['std_train_score']    
        
    means_val = clf.cv_results_['mean_test_score']
    stds_val = clf.cv_results_['std_test_score'] 
    
    #Prediction on a pseudo test set (split from Dataset A) using the best estimator
    ypred, ypred_prob, bulk_runtime, avg_runtime = bulk_runtime_estimation(clf.best_estimator_, xtest, 'other')
    ytrue, ypred = ytest, ypred

    scores_on_best_para = {}   
    print()
    print('-------------------------------------------------')
    print("RF Grid search CV on Dataset A:")
    print()
    print("Training scores:")
    print()
    #print training scores
    for mean, std, params in zip(means_train, stds_train, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
        if params == clf.best_params_:
            scores_on_best_para['training'] = [mean, std]
            
    print()
    
    #print validation scores
    print("Validation scores:")
    print()
    #print grid search validation table
    for mean, std, params in zip(means_val, stds_val, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
        if params == clf.best_params_:
            scores_on_best_para['validation'] = [mean, std]
    print()
    

    print("Prediction on a pseudo test set (split from Dataset A):")
    print(classification_report(ytrue, ypred))
    pseudo_test_score = accuracy_score(ytrue, ypred)
    scores_on_best_para['pseudo_test'] = pseudo_test_score
    print("Accuracy:", pseudo_test_score)
    print()
    print("Best parameters found on Dataset A:")
    print()
    print(clf.best_params_)
    print()
    print('Average runtime per test instance:', avg_runtime)
    print('-------------------------------------------------')
    print()
    print()
    
    
    #determine important features
    importances = clf.best_estimator_.feature_importances_
    indices = np.argsort(importances)
    #plt.title('Feature Importances')
    #plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    #plt.yticks(range(len(indices)), [i for i in indices])
    #plt.xlabel('Relative Importance')
    #plt.show()
    
    
    cat_ytrue = keras.utils.to_categorical(ytest)
    # confusion matrix
    plot_confusion_matrix(cat_ytrue, ypred_prob, 'Random Forest', savepath)
    # ROC curve
    plot_ROC_curve(cat_ytrue, ypred_prob, num_class, 'Random Forest', savepath)
    #plot learning curve for best para combination
    lctrain_sizes, lctrain_scores, lcvalid_scores = learning_curve(clf.best_estimator_, xtrain, ytrain, train_sizes=np.linspace(0.1, 1.0, 5), cv=cvfold)
    plot_learning_curves_other(lctrain_sizes, lctrain_scores,lcvalid_scores, 'Random Forest', savepath)

    #return classifier
    return clf.best_estimator_, scores_on_best_para, indices
    
    


def KNN_Grid_search_Prediction_CV(xtrain, ytrain, xtest, ytest, tuning_parameters, cvfold, num_class, savepath):
    clf = GridSearchCV(KNeighborsClassifier() , tuning_parameters, cv = cvfold, return_train_score = True)
    clf.fit(xtrain, ytrain)
    
    means_train = clf.cv_results_['mean_train_score']
    stds_train  = clf.cv_results_['std_train_score']    
        
    means_val = clf.cv_results_['mean_test_score']
    stds_val = clf.cv_results_['std_test_score'] 
    
    #Prediction on a pseudo test set (split from Dataset A) using the best estimator
    ypred,ypred_prob, bulk_runtime, avg_runtime = bulk_runtime_estimation(clf.best_estimator_, xtest, 'other')
    ytrue, ypred = ytest, ypred
    scores_on_best_para = {}

    print()
    print('-------------------------------------------------')
    print("KNN Grid search CV on Dataset A:")
    print()
    print("Training scores:")
    print()
    #print training scores
    for mean, std, params in zip(means_train, stds_train, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
        if params == clf.best_params_:
            scores_on_best_para['training'] = [mean, std]

    print()
    
    #print validation scores
    print("Validation scores:")
    print()
    #print grid search validation table
    for mean, std, params in zip(means_val, stds_val, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
        if params == clf.best_params_:
            scores_on_best_para['validation'] = [mean, std]

    print()
    

    print("Prediction on a pseudo test set (split from Dataset A):")
    print(classification_report(ytrue, ypred))
    pseudo_test_score = accuracy_score(ytrue, ypred)
    scores_on_best_para['pseudo_test'] = pseudo_test_score
    print("Accuracy:", pseudo_test_score)
    print()
    print("Best parameters found on Dataset A:")
    print()
    print(clf.best_params_)
    print()
    print('Average runtime per test instance:', avg_runtime)
    print('-------------------------------------------------')
    print()
    print()
    
    cat_ytrue = keras.utils.to_categorical(ytest)
    # confusion matrix
    plot_confusion_matrix(cat_ytrue, ypred_prob, 'KNN', savepath)
    # ROC curve
    plot_ROC_curve(cat_ytrue, ypred_prob, num_class, 'KNN', savepath)
    #plot learning curve for best para combination
    lctrain_sizes, lctrain_scores, lcvalid_scores = learning_curve(clf.best_estimator_, xtrain, ytrain, train_sizes=np.linspace(0.1, 1.0, 5), cv=cvfold)
    plot_learning_curves_other(lctrain_sizes, lctrain_scores,lcvalid_scores, 'KNN', savepath)
    
    #return classifier
    return clf.best_estimator_, scores_on_best_para
    
    

    


def Sequential_Prediction_GridSearch_CV(xtrain, ytrain, xtest, ytest, grid_parameters, cvfold, num_class, num_epochs, savepath):
    
    learning_curve_dict_all_models = {}
    
    acc_all_models = []
    acc_std_all_models = []
    loss_all_models = []
    runtime_mean_all_models = []
    
    all_models = []
    
    #the para combination index in search grid
    model_idx = 0
    
    for model_parameters in grid_parameters:
        
        print()
        
        print('########################################')
        model, acc_mean, acc_std_mean, loss_mean, runtime_mean, learning_curve_dict = Sequential_Prediction_CV(xtrain, ytrain, xtest, ytest, model_parameters, cvfold, num_class, num_epochs, model_idx, savepath)
        
        #store returned values for each model
        acc_all_models.append(acc_mean)
        acc_std_all_models.append(acc_std_mean)
        loss_all_models.append(loss_mean)
        runtime_mean_all_models.append(runtime_mean)
        
        all_models.append(model)
        
        #store learning and loss curve
        learning_curve_dict_all_models[model_idx] = learning_curve_dict
        

        print('########################################')
        
        model_idx = model_idx + 1
        
        
       
        
    #find max acc value on pseudo test set and index
    max_acc_value = max(acc_all_models)
    max_acc_value_index = acc_all_models.index(max_acc_value)
    
    
    
    #plot learning and loss curve
    plot_avg_learning_curves_all_models(learning_curve_dict_all_models, 'accuracy', savepath)
    plot_avg_learning_curves_all_models(learning_curve_dict_all_models, 'loss', savepath)
    
        
        
        
    
    print('Opt model parameter found on the pseudo test set:')
    print()
    print(grid_parameters[max_acc_value_index])
    print()
    print('Best average pseudo test set accuracy score with the opt model: {}'.format(max_acc_value))
    print('Average runtime per test instance: {}'.format(runtime_mean_all_models[max_acc_value_index]))
    
    
    
    return all_models[max_acc_value_index], max_acc_value
    
    
    
  


def Sequential_Prediction_CV(xtrain, ytrain, xtest, ytest, model_parameters, cvfold, num_class, num_epochs, model_idx, savepath):
    
    #a dict to store learning curve and loss curves
    learning_curve_dict = {}
    
    # aggregate inputs and targets
    #inputs = np.concatenate((xtrain, xtest), axis=0)
    #targets = np.concatenate((ytrain, ytest), axis=0)
    inputs  = xtrain
    targets = ytrain
    
    fold_no = 1
    #instead of using GridsearchCV we apply a manual function
    kf = KFold(n_splits=cvfold, random_state=None, shuffle=True)
    #record accuracy and loss per fold
    acc_per_fold  = []
    loss_per_fold = []
    avg_runtime_per_fold = []

    #average history train and val acc across all folds
    avg_history_train_acc = np.zeros(num_epochs)
    avg_history_val_acc = np.zeros(num_epochs)
    avg_history_train_loss = np.zeros(num_epochs)
    avg_history_val_loss = np.zeros(num_epochs)
    
    #an empty array to store confusion matrix every fold
    cm_sum = np.zeros((num_class,num_class))
    
    #input feature shape 
    in_feature_shape = xtrain.shape[1]
    #number of class (labels)
    num_labels = num_class
    
    #kf.split divide inputs and targets into K folds and
    #train_idx, test_idx are the indices for training set and validation set
    for train_idx, test_idx in kf.split(inputs, targets):
        
        print('------------------------------------------------------------------------')
        print('Squential with Kfold CV: ')
        print('Training for fold {} ...'.format(fold_no))
        
        #reconstruct model for each fold
        model = design_sequential_model(model_parameters, in_feature_shape, num_class)

        #
        history = model.fit(inputs[train_idx], targets[train_idx],
                    batch_size=int(xtrain.shape[0]/10),
                    epochs=num_epochs,
                    validation_data=(inputs[test_idx], targets[test_idx]),
                    verbose=1)
        
        
        
        #evaluate pseudo test set (split from Dataset A)
        scores = model.evaluate(xtest, ytest, verbose=1)
                
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        
        
        #save history
        avg_history_train_acc = avg_history_train_acc + history.history['accuracy']
        avg_history_val_acc = avg_history_val_acc + history.history['val_accuracy']
        avg_history_train_loss = avg_history_train_loss + history.history['loss']
        avg_history_val_loss = avg_history_val_loss + history.history['val_loss']
        
        
        #generate predictions
        #ypred = model.predict(xtest)
        #model.predict returns probability
        ypred, ypred_prob, bulk_runtime, avg_runtime = bulk_runtime_estimation(model, xtest, 'seq')
        
        #convert to single value labels
        #ypred_labels = np.argmax(ypred_prob, axis=1)
        #ytest_labels = np.argmax(ytest, axis=1)
        
        #ytrue, ypred = ytest_labels, ypred_labels
        
        avg_runtime_per_fold.append(avg_runtime)
        
        # confusion matrix 
        #plot_confusion_matrix(ytest, ypred_prob)
        cm_sum = cm_sum + confusion_matrix_per_fold(ytest, ypred_prob)        
        
        # ROC curve
        plot_ROC_curve(ytest, ypred_prob, num_class, 'MLP', savepath)
        
        #plot learning curve
        #plot_learning_curve(history, fold_no, model_idx)

        # Increase fold number
        fold_no = fold_no + 1
        
    # == Provide average scores ==
    print()
    mean_acc_per_fold = np.mean(acc_per_fold)
    std_acc_per_fold = np.std(acc_per_fold)
    mean_loss_per_fold = np.mean(loss_per_fold)
    
    mean_runtime_per_fold = np.mean(avg_runtime_per_fold)
    
    
    #plot average confusion matrix
    plot_avg_confusion_matrix_across_folds(cm_sum, cvfold, 'MLP', savepath)
    
    
    #plot avg learning curve accross all folds
    avg_history_train_acc = np.multiply(avg_history_train_acc, 1/cvfold)
    avg_history_val_acc = np.multiply(avg_history_val_acc, 1/cvfold)
    avg_history_train_loss = np.multiply(avg_history_train_loss, 1/cvfold)
    avg_history_val_loss = np.multiply(avg_history_val_loss, 1/cvfold)
    
    
    #plot learning curve
    plot_avg_learning_curve(avg_history_train_acc, avg_history_val_acc, model_idx, 'accuracy', savepath)
    #plot loss curve
    plot_avg_learning_curve(avg_history_train_loss, avg_history_val_loss, model_idx, 'loss', savepath)
    
    
    
    #append into dict
    learning_curve_dict['avg_history_train_acc'] = avg_history_train_acc
    learning_curve_dict['avg_history_val_acc'] = avg_history_val_acc
    learning_curve_dict['avg_history_train_loss'] = avg_history_train_loss
    learning_curve_dict['avg_history_val_loss'] = avg_history_val_loss
    
    

    print('Average scores for pesudo test set across all folds:')
    print('> Accuracy: {} (+- {})'.format(mean_acc_per_fold, std_acc_per_fold))
    print('> Loss: {}'.format(mean_loss_per_fold))
    print('> Avg runtime per test instance: {}'.format(mean_runtime_per_fold))
    print('------------------------------------------------------------------------')
    
    return model, mean_acc_per_fold, std_acc_per_fold, mean_loss_per_fold, mean_runtime_per_fold, learning_curve_dict

    
    
    

#design a sequential model according to input parameter grid
#for multiclass the loss function should be category_crossentropy, not binary
def design_sequential_model(params, inshape, num_class):
    
    model = Sequential()
    num_hidden_layer = params['num_hidden_layer']
    
    # initial layer
    # we use 2^(hiddenlayer +1) as the output number of neuron 
    model.add(Dense(inshape*pow(2, num_hidden_layer+1) , input_shape=(inshape,),
                    activation='relu'))
    
    #If user has specify the output neuron at each hidden layer:
    if "hidden_neuron" in params:
        
        # hidden layers
        for i in range(num_hidden_layer):
            print('Adding layer '+str(i+1)+':')
        
            model.add(Dense(params['hidden_neuron'][i], activation=params['hidden_layer_activation'][i]))
            model.add(Dropout(params['dropout'][i]))
            
    
    #default output neuron at each hidden layer: descending at base 2
    else:
                # hidden layers
        for i in range(num_hidden_layer):
            print('Adding layer '+str(i+1)+':')
        
            model.add(Dense(inshape*pow(2, num_hidden_layer - i), activation=params['hidden_layer_activation'][i]))
            model.add(Dropout(params['dropout'][i]))
            
    
    ## final layer
    model.add(Dense(num_class, activation=params['last_activation']))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    
    return model
 



#credit: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem

def plot_ROC_curve(cat_ytrue, ypred_prob, num_class, model_name, savepath):
    
    #ytest_labels = np.argmax(cat_ytrue, axis=1)
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
    ytrue, ypred = cat_ytrue, ypred_prob
    n_classes = num_class
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    #check if ypred contains any nan. 
    #If nan then the model intrinsically not suitable (i.e. MLP with too few layers)
    checknan = np.isnan(ypred).any()
    if checknan: return 
    
    for i in range(n_classes):

        
        fpr[i], tpr[i], _ = metrics.roc_curve(ytrue[:, i], ypred[:, i]) 
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        

        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(ytrue.ravel(), ypred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'darkviolet'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name +'ROC curve')
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(savepath,'ROC' + '_' + model_name + now + '.png'))
    #plt.show()
    



#This function is used for models other than Sequential:

def plot_confusion_matrix(cat_ytrue, ypred_prob, model_name, savepath):
    
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    

    
    
    #convert category and prob matrix to single value labels
    ypred_labels = np.argmax(ypred_prob, axis=1)
    ytest_labels = np.argmax(cat_ytrue, axis=1)
        
    ytrue, ypred = ytest_labels, ypred_labels
    

    #print(ypred_labels)
    #print(ypred_prob)
    #print('---------')
    #print(ytest_labels)
    #print(cat_ytrue)
    
    cm = confusion_matrix(ytrue, ypred)
    plt.figure()
    plt.matshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    plt.title(model_name + 'Confusion matrix')
    plt.colorbar()
    
    fmt = 'd' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(savepath,'CM' + '_' + model_name + now + '.png'))
    #plt.show()
    
    #return cm
    
    


def confusion_matrix_per_fold(cat_ytrue, ypred_prob):
    
    #convert category and prob matrix to single value labels
    ypred_labels = np.argmax(ypred_prob, axis=1)
    ytest_labels = np.argmax(cat_ytrue, axis=1)
        
    ytrue, ypred = ytest_labels, ypred_labels
    

    cm = confusion_matrix(ytrue, ypred)
    
    
    return cm
    



def plot_avg_confusion_matrix_across_folds(cm_sum, cv_folds, model_name, savepath):
    
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    avg_cm = np.multiply(cm_sum, 1/cv_folds)
    
    #convert to int
    cm = avg_cm.astype(int)
    
    
    plt.figure()
    plt.matshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    plt.title('MLP Avg Confusion matrix across all folds')
    plt.colorbar()
    
    fmt = 'd' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(savepath,'CM' + '_' + model_name + now + '.png'))
    #plt.show()
    

    



# this is for sequential model
def plot_learning_curve(history, fold_no, model_idx, savepath):
    
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy (per fold)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(savepath,'MLP learning_curve' + '_' + 'model'+ str(model_idx) + '_'+ now + '_'+'fold' + 'fold_no'+'.png'))
    #plt.show()
    
    
    

def plot_avg_learning_curve(avg_train_history, avg_val_history,model_idx, ylabel, savepath):
    
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.figure()
    plt.plot(avg_train_history)
    plt.plot(avg_val_history)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(savepath,'MLP learning_curve' + '_' + 'model'+ str(model_idx) + '_'+ now +'.png'))
    #plt.show()
    
    



def plot_avg_learning_curves_all_models(learning_curve_dict_all_models, ylabel, savepath):
    
    
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'darkviolet'])
    linestyle=cycle(['-', '--', ':', '-.'])
    lw = 2
    
    if ylabel == 'accuracy':
        plt.figure()
        for key, linestyle in zip(learning_curve_dict_all_models, linestyle):
        
            avg_history_train_acc = learning_curve_dict_all_models[key]['avg_history_train_acc'] 
            avg_history_val_acc = learning_curve_dict_all_models[key]['avg_history_val_acc'] 
        
            plt.plot(avg_history_train_acc, lw=lw, linestyle = linestyle,
                     label='training acc of MLP para set {}'.format(key))
            plt.plot(avg_history_val_acc, lw=lw, linestyle = linestyle,
                     label='val acc of MLP para set {}'.format(key))

        plt.ylim([0.0, 1.05])
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.title('MLP' +'learning curve all models')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(savepath,'Learningcurve' + '_' + now + '.png'))
        #plt.show()
    
    #plot loss curve
    elif ylabel == 'loss':
        plt.figure()
        for key, linestyle in zip(learning_curve_dict_all_models, linestyle):
 
            avg_history_train_loss = learning_curve_dict_all_models[key]['avg_history_train_loss'] 
            avg_history_val_loss = learning_curve_dict_all_models[key]['avg_history_val_loss'] 
        
            plt.plot(avg_history_train_loss, lw=lw, linestyle = linestyle,
                     label='training loss of MLP para set {}'.format(key))
            plt.plot(avg_history_val_loss, lw=lw, linestyle = linestyle,
                     label='val loss of MLP para set {}'.format(key))

        plt.ylim([0.0, 1.05])
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.title('MLP' +'loss curve all models')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(savepath,'Losscurve' + '_' + now + '.png'))
        #plt.show()
        
    else:
        return
        
    


def plot_learning_curves_other(train_sizes, train_scores, valid_scores, model_name, savepath):
    
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    
    plt.figure()    
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    plt.xlabel('training examples')
    plt.ylabel('score')
    plt.title('Learningcurve'+ model_name)
    plt.savefig(os.path.join(savepath,'Learningcurve' + '_' + now + model_name +'.png'))
    #plt.show()
    

def save_model_to_disk(model, filename, savepath):
    #pickle.dump(model, open(filename, 'wb'))
    fullpath = os.path.join(savepath,filename)
    with open(fullpath, 'wb') as file:
        pickle.dump(model, file)




def convert_true_test_data(landmark_features, label_contents):

    #shuffle
    true_testX, true_testY = shuffle(landmark_features, label_contents)
    
    #reshape into appropriate dimensions
    true_testX = true_testX.reshape((true_testX.shape[0], true_testX.shape[1]*2))
    
    #normalisation
    x_scaler = StandardScaler()
    true_testX = x_scaler.fit_transform(true_testX)
    
    #convert single valued Y into categorical 
    cat_true_testY = keras.utils.to_categorical(true_testY)

    return true_testX, true_testY, cat_true_testY


#this is for clf other than MLP
def true_test_with_opt_clf(te_X, te_Y, optclf):
    #Prediction on a pseudo test set (split from Dataset A) using the best estimator
    ypred, ypred_prob, bulk_runtime, avg_runtime = bulk_runtime_estimation(optclf, te_X, 'other')
    ytrue, ypred = te_Y, ypred

    print()
    print("Prediction on true test set:")
    print(classification_report(ytrue, ypred))
    true_test_score = accuracy_score(ytrue, ypred)
    print("True test set Accuracy: {}".format(true_test_score))
    print()
    print('Average runtime per test instance: {}'.format(avg_runtime))
    print('-------------------------------------------------')

    return  true_test_score 

def true_test_with_opt_seqmodel(te_X, cat_te_Y, Seqclf):

    Seq_true_test_res_score = {}
    #evaluate true test set 
    scores = Seqclf.evaluate(te_X, cat_te_Y, verbose=1)
    Seq_true_test_res_score['true_test_acc'] = scores[1]
    Seq_true_test_res_score['true_test_loss'] = scores[0]

    ypred, ypred_prob, bulk_runtime, avg_runtime = bulk_runtime_estimation(Seqclf, te_X, 'seq')

    print('scores for true test set:')
    print('> Accuracy: {}'.format(scores[1]))
    print('> Loss: {}'.format(scores[0]))
    print('> Avg runtime per test instance: {}'.format(avg_runtime))
    print('------------------------------------------------------------------------')

    return Seq_true_test_res_score





def get_B1_results():

    TaskB1_opt_models_dict = {}
    TaskB1_res_dict = {}

    #create a folder to save generated pickles, clf, images
    savepath = os.path.join('B1', 'TaskB1_res')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)


    print()
    print('-------------------------Reading Task B1 dataset---------------------------')
    #68 feature contents (from image), corresponding label (from csv), the image name where dlib cannot extract features
    landmark_features, label_contents, no_features_sets = B1_extract.extract_features_labels(image_folder, label_name)
    #number of unique classes to be predicted
    num_class = len(np.unique(label_contents))

    tr_X, te_X, tr_Y, te_Y, cat_tr_Y, cat_te_Y  = split_data(landmark_features, label_contents, split_ratio)

    #save read dataset for future use
    save_model_to_disk(landmark_features, "TaskB1_landmark_features.pickle", savepath)
    save_model_to_disk(label_contents, "TaskB1_label_contents.pickle", savepath)
    save_model_to_disk(no_features_sets, "TaskB1_no_features_sets.pickle", savepath)


    #search through the parameter grid
    #the returned estimator : Logclf, SVMclf, RFclf, KNNclf and Seqmodel,
    #are trained clf (with celeba and cartoon_set) with best parameter combination through search

    # SVM result:
    print()
    print('-------------------------Task B1: Gird searching on SVM---------------------------')
    SVMclf, SVM_scores_on_best_para = SVC_GridSearch_PredictionCV(tr_X, tr_Y, te_X, te_Y, SVM_tuned_parameters, cvfold, num_class, savepath)
    TaskB1_opt_models_dict['SVM'] = SVMclf
    TaskB1_res_dict['SVM'] = SVM_scores_on_best_para
    print()

    # Random Forest result:
    print()
    print('-------------------------Task B1: Gird searching on Rando Forest---------------------------')
    RFclf, RF_scores_on_best_para, RF_importance_indices = RF_GridSearch_PredictionCV(tr_X, tr_Y, te_X, te_Y, RF_tuned_parameters, cvfold, num_class, savepath)
    TaskB1_opt_models_dict['RF'] = RFclf
    TaskB1_res_dict['RF'] = RF_scores_on_best_para
    print()

    # KNN resulS:
    print()
    print('-------------------------Task B1: Gird searching on KNN------------------------------')
    KNNclf, KNN_scores_on_best_para = KNN_Grid_search_Prediction_CV(tr_X, tr_Y, te_X, te_Y, KNN_tuned_parameters, cvfold, num_class, savepath)
    TaskB1_opt_models_dict['KNN'] = KNNclf
    TaskB1_res_dict['KNN'] = KNN_scores_on_best_para
    print()

    #Sequential model results:
    print()
    print('-------------------------Task B1: Gird searching on MLP------------------------------')
    Seqmodel, Seq_pseudo_test_score = Sequential_Prediction_GridSearch_CV(tr_X, cat_tr_Y, te_X, cat_te_Y, Squential_model_parameters, cvfold, num_class, num_epochs, savepath)
    TaskB1_opt_models_dict['MLP'] = Seqmodel
    TaskB1_res_dict['MLP'] = Seq_pseudo_test_score
    print()


    #save res for future use (MLP models cannot be pickled so we do not save)
    save_model_to_disk(TaskB1_res_dict, "TaskB1_res_dict.pkl", savepath)

    #save trained models wit optimum para set after searching
    save_model_to_disk(SVMclf, "TaskB1_SVMclf.pkl", savepath)
    save_model_to_disk(RFclf, "TaskB1_RFclf.pkl", savepath)
    save_model_to_disk(KNNclf, "TaskB1_KNNclf.pkl", savepath)

    Seqmodel.save(os.path.join(savepath, "TaskB1_Seqmodel"))




    return TaskB1_opt_models_dict, TaskB1_res_dict





def get_B1_true_test_res(true_test_image_folder, true_test_label_name, model_dict, use_pretrained):

    TaskB1_true_test_res_dict = {}

    loadpath = os.path.join('B1', 'TaskB1_pretrained')
    savepath = os.path.join('B1', 'TaskB1_res')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
        
    print()
    print('-------------------------Reading Task B1 dataset---------------------------')
    #68 feature contents (from image), corresponding label (from csv), the image name where dlib cannot extract features
    landmark_features, label_contents, no_features_sets = B1_extract.extract_features_labels(true_test_image_folder, true_test_label_name)
    #number of unique classes to be predicted
    num_class = len(np.unique(label_contents))

    print(label_contents.shape)

    #save read dataset for future use
    save_model_to_disk(landmark_features, "TaskB1_true_landmark_features.pickle", savepath)
    save_model_to_disk(label_contents, "TaskB1_true_label_contents.pickle", savepath)
    save_model_to_disk(no_features_sets, "TaskB1_true_no_features_sets.pickle", savepath)

    #normaisation and convert to categorical matrix
    true_testX, true_testY, cat_true_testY = convert_true_test_data(landmark_features, label_contents)

    #use pretrained models
    if use_pretrained == True:
        #load models
        SVMclf = load_model_from_disk("TaskB1_SVMclf.pkl", loadpath, 'pickle')
        RFclf = load_model_from_disk("TaskB1_RFclf.pkl", loadpath, 'pickle')
        KNNclf = load_model_from_disk("TaskB1_KNNclf.pkl", loadpath, 'pickle')
        Seqmodel = load_model_from_disk("TaskB1_Seqmodel", loadpath, 'Seq')

    #import model from previous step
    else:     
        #opt models from grid search:
        SVMclf = model_dict['SVM']
        RFclf = model_dict['RF']
        KNNclf = model_dict['KNN']
        Seqmodel = model_dict['MLP']



    print()
    print('-------------------------Task B1: True test performance with SVM------------------------------')
    SVM_true_test_score = true_test_with_opt_clf(true_testX, true_testY, SVMclf)
    TaskB1_true_test_res_dict['SVM'] = {}
    TaskB1_true_test_res_dict['SVM']['true_test_score'] = SVM_true_test_score

    print()
    print('-------------------------Task B1: True test performance with RF------------------------------')
    RF_true_test_score = true_test_with_opt_clf(true_testX, true_testY, RFclf)
    TaskB1_true_test_res_dict['RF'] = {}
    TaskB1_true_test_res_dict['RF']['true_test_score'] = RF_true_test_score

    print()
    print('-------------------------Task B1: True test performance with KNN------------------------------')
    KNN_true_test_score = true_test_with_opt_clf(true_testX, true_testY, KNNclf)
    TaskB1_true_test_res_dict['KNN'] = {}
    TaskB1_true_test_res_dict['KNN']['true_test_score'] = KNN_true_test_score

    print()
    print('-------------------------Task B1: True test performance with MLP------------------------------')
    Seq_true_test_res_score = true_test_with_opt_seqmodel(true_testX, cat_true_testY, Seqmodel)
    TaskB1_true_test_res_dict['MLP'] = {}
    TaskB1_true_test_res_dict['MLP']['true_test_score'] = Seq_true_test_res_score 


    save_model_to_disk(TaskB1_true_test_res_dict, "TaskB1_true_test_res_dict.pkl", savepath)

    return TaskB1_true_test_res_dict






def load_model_from_disk(filename, savepath, typename):
    fullpath = os.path.join(savepath,filename)

    #saved by pickle
    if typename == 'pickle':
        with open(fullpath, 'rb') as file:
            model = pickle.load(file)

    elif typename == 'Seq':
            model = keras.models.load_model(fullpath)
    return model





