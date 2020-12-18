# AMLS_20-21_SN15105271

Student ID: 15105271  

Module code: ELEC0134 Applied machine learning system 20/21 assignment  

email address: zczliue@ucl.ac.uk

```diff
- This repository uses [Git LFS](https://git-lfs.github.com) to handle larger files such as pretrained models

- It must be cloned with Git lfs installed via command line 
    git lfs clone https://github.com/zczliue/AMLS2021SN15105271.git
    
- Otherwise, files may be corrupted if using other methods (such as directly clone from browser or git clone in command line)

- Please leave sufficient disk storage (at least 10GB), or otherwise PIL may report error when reading images
```

## Table of Contents 
- [Introduction](#Introduction)
- [Files](#Files)
- [DataSplitting](#DataSplitting)
- [UserGuide](#UserGuide)
- [ErrorHandling](#ErrorHandling)
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
Explanation for the files stored in repository main branch:
- [A1](./A1) (folder): 
  - A1_functions_all.py: function file that carries out grid search, training, testing and image plots
  - A1_extract_landmarks.py: function file that adapted from AMLS lab2 to read and extract 68 facial landmarks and labels with dlib library
  - TaskA1_pretrained (folder): Stores pretrained models.
  - TaskA1_res (folder): This folder is none by default and will be created during coderun to store generated plots, extracted landamrks pickles, re-trained models. 
  
 - A2 and B1 (folder): same as A1 folder
 
 - [B2](./B2) (folder) additional :
    - B2_preprocess.py: function file that extract dominant rgb value for eye area. Therefore instead of using 68 features,  B2_functions_all.py uses 3 values of rgb as input features.

- [main_results_Jupyter.pdf](./main_results_Jupyter.pdf): A Jupyter report printing the results obtained 

- [main.py](./main.py): main function to print result summary

- [Prerequisites_full_list.txt](./Prerequisites_full_list.txt) : List of all packages and versions required print by sys and pycharm

- Datasets: empty by default


## DataSplitting
The testbed defines 2 categories:
- Category A: celeba (5000 imgs) and cartoon_set (10000 imgs)
  - This dataset is used as 0.75 train-test split and the train set is further applied with a 5-fold cross validation during grid search for each model. The test score returned is termed as a (pseudo) test score.

- Category B: celeba_test (1000 imgs) and cartoon_set_test (2500 imgs)
  - This dataset is used as the true test set to be predicted by the optimal models trained with Dataset A. The true test score will be compared with the pseudo test score to see if they agree with each other.
  

 

## UserGuide


```diff
- This repository uses [Git LFS](https://git-lfs.github.com) to handle larger files such as pretrained models

- It must be cloned with Git lfs installed via command line 
    git lfs clone https://github.com/zczliue/AMLS2021SN15105271.git
    
- Otherwise, files may be corrupted if using other methods (such as directly clone from browser or git clone in command line)
```

To use this testbed, 

- install Git LFS and clone via **git lfs clone url**
- load celeba, cartoon_set, celeba_test, cartoon_set_test under Datasets folder
- in main branch, run python3 main.py 


The main function contains 3 sections: 

- Sec 1: Generate true test score with pretrained models
  - This section loads pretrained models with optimal parameter combinations for each model type (stored in <TaskNumber_pretrained> folder within each task), and then perform predictions on the true test Dataset B (celeba_test and cartoon_set_test).
   - The estimated runtime for each task is:
      - Task A1: 39 sec
      - Task A2: 37 sec
      - Task B1: 337 sec
      - Task B2: 6156 sec
```diff
- The entire Section 1 can be commented out without affecting the coderun of the next 2 sections. 
- Please skip this section if any pretrained model is corrupted due to improper LFS management. 
 ```
 
- Sec 2: Grid search and generate train, validate and (pseudo) test score 
  - This section loads Category A (celeba and cartoon_set) and performs grid search CV on each model type. Generated plots, intermediate data pickles and newly trained models will be stored in <TaskNumber_res> folder within each task.
  
  - For each task, the main function calls TaskNumber_functions_all.py to go through the 5 model types in sequence. Then, TaskNumber_functions_all.py calls TaskNumber_extract_landmarks.py to extract features and return to TaskNumber_functions_all.py
  
  - The estimated runtime for each task is:
    - Task A1: 4261 sec 
    - Task A2: 3748 sec
    - Task B1: 22044 sec
    - Task B2: 23703 sec (the majority of runtime is spent on preprocessing the image)
     
  - Alternatively, the user may also change grid search parameter range by changing the header of <TaskNumber_functions_all.py> under each task folder. The current search range is same for all 4 tasks, given as:

    - Logistic Regression: 'solver':[ 'saga'], 'penalty':['l1', 'l2'], 'C':[1e-4, 1e-3, 1e-2, 1e-1, 1,10]
    - SVM: 'kernel': ['rbf', 'linear'], 'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1]
    - RF: 'n_estimators':[64, 128, 256, 512, 1024], 'max_depth':[4, 8, 16, 32, 64]
    - KNN: 'n_neighbors':[8,16,32,64,128]
    - MLP: num_epochs = 20, batch_size = Num_train_samples/10
      - Para set 0: {'num_hidden_layer': 3, 'hidden_layer_activation' : ['relu','tanh','relu'], 'dropout':[0.5,0.25,0.125], 'last_activation':'softmax'}
      - Para set 1: {'num_hidden_layer': 4, 'hidden_layer_activation' : ['relu','tanh','relu','tanh'],'dropout':[0.5,0.25,0.125,0.0625],'last_activation':'softmax'}

- Sec 3: Generate true test score with re-trained models
  - This section loads Category B (celeba_test and cartoon_set_test) and performs prediction with newly trained models generated in Sec 2. Intermediate data pickles are stored in <TaskNumber_res> folder within each task.
 
 
 
 ## ErrorHandling
 
 - Insufficient disk storage: 
    - img = pil_image.open(io.BytesIO(f.read())) 
    - "cannot identify image file %r" % (filename if filename else fp) 
    - PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x13d45fa40>:
        - These errors might arise from lack of disk storage when attempting to read the Datasets.Please leave at least 10GB to run the code. 
    
 - Corruption of pretrained models
    - The entire Section 1 can be commented out without affecting the coderun of the next 2 sections. Please skip this section if any pretrained model is corrupted due to improper LFS management. 
 
 
 
 
## Prerequisites


- **Git LFS**
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

Please also refer to [Prerequisites_full_list.txt](./Prerequisites_full_list.txt) for full list of package versions. 


