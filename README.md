# AMLS_ELEC0134_20_21

Student ID: 15105271  
Mres Connected Electronic and Photonic Systems
Module code: ELEC0134 Applied machine learning system 20/21 assignment  

## Table of Contents 
- [Introduction](#Introduction)
- [Files](#Files)
- [DataSplitting](#DataSplitting)
- [UserGuide](#UserGuide)
- [Prerequisites](#Prerequisites)


## Introduction
This is the GitHub repository for ELEC0134 AMLS assignment. The assignment provides a testbed for 4 tasks in Mac OS system including:
- Task A1: gender detection
- Task A2: emotion detection
- Task B1: face shape detection
- Task B2: eye colour detection

For each task, the testbed will run through a grid search with cross validation process for 5 different model types: 
- Logistic regression
- SVM
- Random Forest
- KNN
- MLP dense sequential network

Based on grid search scores, the testbed will select the optimal parameter combination under each model type and compare between their performances in terms of true test accuracy, learning/lost curves, ROC, confusion matrix and runtime.  


## Files
Explanation for the files stored in repository:
- [A1](./A1) (folder): 
  - A1_functions_all.py: function file that carries out grid search, training, testing and image plots
  - A1_extract_landmarks.py: function file that adapted from AMLS lab2 to read and extract 68 facial landmarks and labels with dlib library
  - TaskA1_pretrained (folder): Stores pretrained models.
  - TaskA1_res (folder): This folder is empty (or none) by default and will be used to store generated plots, extracted landamrks pickles, re-trained models as the testbed runs the code. 
  
 - A2 and B1 (folder): same as A1 folder
 
 - [B2](./B2) (folder):
    - B2_preprocess.py: function file that extract dominant rgb value for eye area. Therefore instead of using 68 features,  B2_functions_all.py uses 3 values of rgb as input features.

- main_results_Jupyter.pdf: A Jupyter report printing the results obtained 

- Datasets: empty by default


## DataSplitting
The testbed defines 2 categories:
- Category A: celeba (5000 imgs) and cartoon_set (10000 imgs)
  - This dataset is used as 0.75 train-test split and the train set is further applied with a 5-fold cross validation during grid search for each model. The test score returned is termed as a (pseudo) test score.

- Category B: celeba_test (1000 imgs) and cartoon_set_test (2500 imgs)
  - This dataset is used as the true test set to be predicted by the optimal models trained with Dataset A. The true test score will be compared with the pseudo test score to see if they aggree with each other.
  

 

## UserGuide

To use this testbed, 
- load celeba, cartoon_set, celeba_test, cartoon_set_test under Datasets folder
- run main.py 


The main function contains 3 sections: 

- Sec 1: Generate true test score with pretrained models
  - This section loads pretrained models with optimal parameter combinations for each model type (stored in <TaskNumber_pretrained> folder within each task), and then perform predictions on the true test Dataset B (celeba_test and cartoon_set_test).
  - Whether this section is commented out does not affect the coderun for the following 2 sections. The pretrained models will not be overwritten when the following 2 sections generate newly trained models.
   - The estimated runtime for each task is:
    - Task A1: 39 sec
    - Task A2: 37 sec
    - Task B1: 337 sec
    - Task B2: 6156 sec
  
- Sec 2: Grid search and generate train, validate and (pseudo) test score 
  - This section loads Category A (celeba and cartoon_set) and performs grid search CV on each model type. Generated plots, intermediate data pickles and newly trained models will be stored in <TaskNumber_res> folder within each task.
  - The estimated runtime for each task is:
    - Task A1: 2h
    - Task A2: 2h
    - Task B1: 10h
    - Task B2: 10h (the majority of runtime is spent on preprocessing the image)
     
  - Alternatively, the user may also change grid search parameter range by changing the header of <TaskNumber_functions_all.py> under each task folder. The current search range is same for all 4 tasks, given as:

    - Logistic Regression: 'solver':[ 'saga'], 'penalty':['l1', 'l2'], 'C':[1e-4, 1e-3, 1e-2, 1e-1, 1,10]
    - SVM: 'kernel': ['rbf', 'linear'], 'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1]
    - RF: 'n_estimators':[64, 128, 256, 512, 1024], 'max_depth':[64, 128, 256, 512, 1024]
    - KNN: 'n_neighbors':[8,16,32,64,128]
    - MLP: num_epochs = 50, batch_size = Num_train_samples/10
      - Para set 0: {'num_hidden_layer': 3, 'hidden_layer_activation' : ['relu','tanh','relu'], 'dropout':[0.5,0.25,0.125], 'last_activation':'softmax'}
      - Para set 1: {'num_hidden_layer': 4, 'hidden_layer_activation' : ['relu','tanh','relu','tanh'],'dropout':[0.5,0.25,0.125,0.0625],'last_activation':'softmax'}

- Sec 3: Generate true test score with re-trained models
  - This section loads Category B (celeba_test and cartoon_set_test) and performs prediction with newly trained models generated in Sec 2. Intermediate data pickles are stored in <TaskNumber_res> folder within each task.
 
## Prerequisites
  
- Python 3.6.7
- Mac OS Sierra 10.12.6
- tensorflow 2.3.1
- tensorflow.python.keras 2.4.0
- csv 1.0
- keras_preprocessing 1.1.2
- numpy 1.18.5
- scipy 1.5.4
- matplotlib 3.3.2
- keras 2.4.3
- sklearn 0.21.2
- cv2 4.4.0
- dlib 19.21.0
- pandas 1.1.4
- matplotlib 3.3.2
- json 2.0.9
- itertools 
- datetime
- pickle
- collections
- math
- time
- os

Please also refer to Prerequisites_full_list.txt (./Prerequisites_full_list.txt) for full list of versions. 


