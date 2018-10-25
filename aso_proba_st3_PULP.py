import pandas as pd
import numpy as np
from pulp import *
import pickle
import time
import os
import lib.fcns_asstopt_PULP as fcns_asstopt


print("Optimization of the assortment given a choice model")

min_capacity = 0
max_capacity = 7

algo_chosen = 'GDT'
data_version =  '096'  #096 
    
filename_transaction        = 'transaction_data_bal_'+  data_version + '.dat'
filename_choice_model_GDT   = 'choice_model_GDT_bal_PULP_'+  data_version + '.dat'

filename_aso  = 'aso_bal_'+  data_version + '.dat'
filename_output  = 'output_bal_'+  data_version + '.dat'

script_dir  = '/home/oleh/assortment_optimization-master/sample'
#os.path.dirname(__file__) 
#'/home/oleh/assortment_optimization-master/sample'
rel_path_transaction = "data/"+filename_transaction
abs_file_transaction = os.path.join(script_dir, rel_path_transaction)

rel_path_choice_model_GDT = "data/"+filename_choice_model_GDT
abs_file_choice_model_GDT = os.path.join(script_dir, rel_path_choice_model_GDT)

rel_path_aso = "data/"+filename_aso
abs_file_aso = os.path.join(script_dir, rel_path_aso)

rel_path_output = "data/"+filename_output
abs_file_output = os.path.join(script_dir, rel_path_output)

with open(abs_file_transaction, 'rb') as sales:
    my_depickler = pickle.Unpickler(sales)
    Inventories =     my_depickler.load()
    Proba_product =   my_depickler.load()
    Revenue =         my_depickler.load()
    dic =             my_depickler.load()
    Rev_Baseline =    my_depickler.load()
    Prod_List_Max =   my_depickler.load()
    real_revenue =    my_depickler.load()
    predicted_rev_data = my_depickler.load()
    products_data = my_depickler.load()
    ass_data = my_depickler.load()
    rev_all_products = my_depickler.load()
    max_ass_num_manual = my_depickler.load()

if(algo_chosen=='GDT'):    
    print("Opening choice model, format GDT")
    with open(abs_file_choice_model_GDT, 'rb') as sales:
        my_depickler = pickle.Unpickler(sales)
        sigma_GDT_sorted =   my_depickler.load()
        lambda_GDT_sorted =   my_depickler.load()    
else:
    print("Error; wrong input parameter, which algorithm do you wish to use?")
    
(nb_asst, nb_prod) = Inventories.shape   

if(algo_chosen=='GDT' or algo_chosen=='gen' ):
    Lambda = lambda_GDT_sorted
    Sigma = sigma_GDT_sorted  
    t1 = time.time()
    [x_found, obj_val] = fcns_asstopt.run_asstopt_GDT(Lambda, Sigma, Revenue[:len(Sigma.T)], min_capacity, max_capacity)
    t2 = time.time()

d = []   
for el in dic:
    d.append(el)  

aso_result = []
        
for val in enumerate(d):
    if (val[0] in x_found):
       aso_result.append(val[1][1])

same_prod= np.intersect1d(Prod_List_Max,  aso_result)
diff_prod = np.setdiff1d(aso_result, Prod_List_Max)   
exc_prod = np.setdiff1d(Prod_List_Max , same_prod)


    






