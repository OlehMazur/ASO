import os
import numpy as np
from random import randint
import pickle
import time
import lib.gen_GDT_PULP as gen_GDT
import lib.gen_BM as gen_BM
import sys


print("################################# File learn_choice_model.py #################################")
print("Learning of the choice model")

NB_ITER = 20  #20  #if eps_stop is different than 0, NB_ITER ignored
eps_stop= 0
data_version = '096'
algo_chosen  = 'GDT'  #'GDT'

#  Importation of the generated sales data
filename_transaction = 'transaction_data_bal_'+data_version+'.dat'
#ensures that the path is correct for data file
script_dir = '/home/oleh/assortment_optimization-master/sample'
#os.path.dirname(__file__) #absolute dir the script is in
rel_path_transaction = "data/"+filename_transaction
abs_file_transaction = os.path.join(script_dir, rel_path_transaction)

with open(abs_file_transaction, 'rb') as sales:
    my_depickler = pickle.Unpickler(sales)
    Inventories = my_depickler.load()
    Proba_product =   my_depickler.load()
    Revenue =         my_depickler.load()
    
 
filename_choice_model_GDT   = 'choice_model_GDT_bal_PULP_'+str(data_version)+'.dat'
rel_path_choice_model_GDT = "data/"+filename_choice_model_GDT
abs_file_choice_model_GDT = os.path.join(script_dir, rel_path_choice_model_GDT)

filename_choice_model_BM    = 'choice_model_BM_' +str(data_version)+'.dat'
rel_path_choice_model_BM = "data/"+filename_choice_model_BM
abs_file_choice_model_BM = os.path.join(script_dir, rel_path_choice_model_BM)


if(algo_chosen=='GDT' or algo_chosen=='gen'):
    print("GDT algorithm chosen")
    t1=time.time()
    [sigma_GDT_sorted, lambda_GDT_sorted, obj_val_master, history_obj_val] = \
    gen_GDT.run_GDT(Inventories, Proba_product, NB_ITER, eps_stop=eps_stop)
    t2=time.time()
elif(algo_chosen=='BM'):
    print("BM algorithm chosen")
    t1=time.time()
    [sigma_BM_sorted, lambda_BM_sorted, obj_val_master, history_obj_val] =  \
    gen_BM.run_BM(Inventories, Proba_product, NB_ITER, eps_stop)
    t2=time.time()
else:
    print("Error; wrong input parameter, which algorithm do you wish to use?")    

print("History of objective values:")
print(history_obj_val)
  
if(algo_chosen=='GDT' or algo_chosen=='gen'):
    print("Saving choice model, format GDT")
    with open(abs_file_choice_model_GDT, 'wb') as sales:
        my_pickler = pickle.Pickler(sales,protocol=2)
        my_pickler.dump(sigma_GDT_sorted)
        my_pickler.dump(lambda_GDT_sorted)
        
elif(algo_chosen=='BM'):
    print("Saving choice model, format BM")
    with open(abs_file_choice_model_BM, 'wb') as sales:
        my_pickler = pickle.Pickler(sales,protocol=2)
        my_pickler.dump(sigma_BM_sorted)
        my_pickler.dump(lambda_BM_sorted)
else:
    print("Error; wrong input parameter, which algorithm do you wish to use?")

print("Learning completed in ", t2-t1, "seconds.")
print("Choice model file has been saved in /sample/data/.")










