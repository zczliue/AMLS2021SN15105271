

# Student Number 15105271
# AMLS-20/21 Assignment main function

import sys
import time
from A1 import A1_functions_all as A1_functions
from A2 import A2_functions_all as A2_functions
from B1 import B1_functions_all as B1_functions
from B2 import B2_functions_all as B2_functions



def print_summary(opt_models_dict, res_dict, true_test_res_dict, taskname):

    print()
    print()
    print('--------------------------------------{}----------------------------------'.format(taskname))
    
    for key in res_dict:
        if key == 'MLP':
            print()
            print('--------{}-------'.format(key))
            print(opt_models_dict[key].summary())
            print('Pseudo test score: {}'.format(res_dict[key]))
            print('True test score: {}'.format(true_test_res_dict[key]['true_test_score']))
            print()
        else:
            print()
            print('--------{}-------'.format(key))
            print(opt_models_dict[key])
            print('Training Acc: {} +- {}'.format(res_dict[key]['training'][0], res_dict[key]['training'][1]))
            print('Val Acc: {} +- {}'.format(res_dict[key]['validation'][0], res_dict[key]['validation'][1]))
            print('Pseudo Test Acc: {}'.format(res_dict[key]['pseudo_test']))
            print('True test Acc: {}'.format(true_test_res_dict[key]['true_test_score']))
            print()
    print('--------------------------------------------------------------------------')
    print()
    print()
        
    
def print_module_version():
    for module in sys.modules:
        try:
            print(module,sys.modules[module].__version__)
        except:
            try:
                if  type(sys.modules[module].version) is str:
                    print(module,sys.modules[module].version)
                else:
                    print(module,sys.modules[module].version())
            except:
                try:
                    print(module,sys.modules[module].VERSION)
                except:
                    pass




def main():

    #runtime estimation for each task
    bulk_runtime_task = {}


    #===========================================================================================================
    #------------------------------Print package versions:--------------------------------
    print()
    print('#------------------------------Print package versions:--------------------------------')
    print_module_version()
    print()

    
    

    #===========================================================================================================
    # Use pretrained models to generate test results on <celeba_test> and <cartoon_set_test>
    # The entire Section 1 can be commented out without affecting the coderun of the next 2 sections. 
    # Please skip this section if any pretrained model is corrupted due to improper LFS upload. 
    #
    print()
    print('#------------------------------Sec 1. Generate true test score with pretrained models--------------------------------')
    bulk_runtime_task['true_test_with_pretrained'] = {}

    # we can use pre-trained models, 
    # set use_pretrained = True 
    # this will read corresponding opt models for each model type from Task result folder 

    #A1
    start = time.time()
    TaskA1_true_test_res_dict = A1_functions.get_A1_true_test_res('celeba_test', 'gender', model_dict = None, use_pretrained = True)
    bulk_runtime = time.time() - start
    bulk_runtime_task['true_test_with_pretrained']['A1'] = bulk_runtime

    #A2
    start = time.time()
    TaskA2_true_test_res_dict = A2_functions.get_A2_true_test_res('celeba_test', 'smiling', model_dict = None, use_pretrained = True)
    bulk_runtime = time.time() - start
    bulk_runtime_task['true_test_with_pretrained']['A2'] = bulk_runtime

    #B1
    start = time.time()
    TaskB1_true_test_res_dict = B1_functions.get_B1_true_test_res('cartoon_set_test', 'face_shape', model_dict = None, use_pretrained = True)
    bulk_runtime = time.time() - start
    bulk_runtime_task['true_test_with_pretrained']['B1'] = bulk_runtime

    #B2
    start = time.time()
    TaskB2_true_test_res_dict = B2_functions.get_B2_true_test_res('cartoon_set_test', 'eye_color', model_dict = None, use_pretrained = True)
    bulk_runtime = time.time() - start
    bulk_runtime_task['true_test_with_pretrained']['B2'] = bulk_runtime


    for key in bulk_runtime_task:
        print()
        print('Runtime Estimation for Section {}:'.format(key))
        print('{} seconds'.format(bulk_runtime_task[key]))
        print()




    #======================================================================================================
    print()
    print('-----------------Sec 2. Grid search and generate train, validate and (pseudo) test score ---------------------')

    bulk_runtime_task['model_training'] = {}
    # The following 4 functions each corresponds to the results for 4 tasks and training and validation stage.
    # We use celeba (5000 imgs) and cartoon_set (10000 imgs) to perform a 0.75 train-test split and 5 fold 
    # cross validation within train set.

    # Each task will be tested with 5 different types of models:
    # Logistic Regression, SVM, Random Forest, KNN and MLP 
    # Each model will go through a grid search CV from a list of parameters to select the opt para combination.

    # return values:
    # Task_opt_models_dict: a dict that stores the opt classifiers from grid search for each model category
    # TaskA1_res_dict: the training, vaidation, and (pesudo) test score with opt model

    # Estimated runtime: 
    # Task A1: 3h
    # Task A2: 3h
    # Task B1: 10h
    # Task B2: 10h (the majority of runtime is spent on preprocessing the image)

    # The generated models, imgs and intermediate data will be saved under 4 results foldes:
    # TaskA1_res, TaskA2_res, TaskB1_res, TaskB2_res

    #A1
    start = time.time()
    TaskA1_opt_models_dict, TaskA1_res_dict = A1_functions.get_A1_results()
    bulk_runtime = time.time() - start
    bulk_runtime_task['model_training']['A1'] = bulk_runtime

    #A2
    start = time.time()
    TaskA2_opt_models_dict, TaskA2_res_dict = A2_functions.get_A2_results()
    bulk_runtime = time.time() - start
    bulk_runtime_task['model_training']['A2'] = bulk_runtime

    #B1
    start = time.time()
    TaskB1_opt_models_dict, TaskB1_res_dict = B1_functions.get_B1_results()
    bulk_runtime = time.time() - start
    bulk_runtime_task['model_training']['B1'] = bulk_runtime


    #B2
    start = time.time()
    TaskB2_opt_models_dict, TaskB2_res_dict = B2_functions.get_B2_results()
    bulk_runtime = time.time() - start
    bulk_runtime_task['model_training']['B2'] = bulk_runtime

    for key in bulk_runtime_task:
        print()
        print('Runtime Estimation for Section {}:'.format(key))
        print('{} seconds'.format(bulk_runtime_task[key]))
        print()


    #============================================================================================================
    print()
    print('--------------Sec 3. Generate true test score with grid search opt models from Sec 2----------------------')

    bulk_runtime_task['gen_true_test_scores_with_new_trained_models'] = {}
    # read from true test dataset celeba_test and cartoon_set_test
    # Predict and generate accuracy report with opt models selected from previous step

    #A1
    start = time.time()
    TaskA1_true_test_res_dict = A1_functions.get_A1_true_test_res('celeba_test', 'gender', TaskA1_opt_models_dict, use_pretrained = False)
    bulk_runtime = time.time() - start
    bulk_runtime_task['gen_true_test_scores_with_new_trained_models']['A1'] = bulk_runtime

    #A2
    start = time.time()
    TaskA2_true_test_res_dict = A2_functions.get_A2_true_test_res('celeba_test', 'smiling', TaskA2_opt_models_dict, use_pretrained = False)
    bulk_runtime = time.time() - start
    bulk_runtime_task['gen_true_test_scores_with_new_trained_models']['A2'] = bulk_runtime 


    #B1
    start = time.time()
    TaskB1_true_test_res_dict = B1_functions.get_B1_true_test_res('cartoon_set_test', 'face_shape', TaskB1_opt_models_dict, use_pretrained = False)
    bulk_runtime = time.time() - start
    bulk_runtime_task['gen_true_test_scores_with_new_trained_models']['B1'] = bulk_runtime 



    #B2
    start = time.time()
    TaskB2_true_test_res_dict = B2_functions.get_B2_true_test_res('cartoon_set_test', 'eye_color', TaskB2_opt_models_dict, use_pretrained = False)
    bulk_runtime = time.time() - start
    bulk_runtime_task['gen_true_test_scores_with_new_trained_models']['B2'] = bulk_runtime 


    for key in bulk_runtime_task:
        print()
        print('Runtime Estimation for Section {}:'.format(key))
        print('{} seconds'.format(bulk_runtime_task[key]))
        print()


    #=======================================================================================
    ## Print out results :
    print_summary(TaskA1_opt_models_dict, TaskA1_res_dict, TaskA1_true_test_res_dict, 'Task A1')

    print_summary(TaskA2_opt_models_dict, TaskA2_res_dict, TaskA2_true_test_res_dict, 'Task A2')

    print_summary(TaskB1_opt_models_dict, TaskB1_res_dict, TaskB1_true_test_res_dict, 'Task B1')

    print_summary(TaskB2_opt_models_dict, TaskB2_res_dict, TaskB2_true_test_res_dict, 'Task B2')





if __name__ == '__main__':
    main()





