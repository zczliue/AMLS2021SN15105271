# AMLS_ELEC0134_20_21

Student ID: 15105271  
Mres Connected Electronic and Photonic Systems
Module code: ELEC0134 Applied machine learning system 20/21 assignment  

## Table of Contents 
- [Introduction](#Introduction)
- [Files](#Files)
- [DataSplitting](#DataSplitting)
- [Prerequisites](#Prerequisites)
- [UserGuide](#UserGuide)


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
The testbed defines 2 category:
- Dataset A: celeba (5000 imgs) and cartoon_set (10000 imgs)
  - This dataset is used as 0.75 train-test split and the train set is further applied with a 5-fold cross validation during grid search for each model. The test score returned is termed as a (pseudo) test score.

- Dataset B: celeba_test (1000 imgs) and cartoon_set_test (2500 imgs)
  - This dataset is used as the true test set to be predicted by the optimal models trained with Dataset A. The true test score will be compared with the pseudo test score to see if they aggree with each other.
  
  
  
## Prerequisites
  
- Python 3.6.7
- Mac OS Sierra 10.12.6
- tensorflow 2.3.1
- tensorflow.python.keras 2.4.0
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



