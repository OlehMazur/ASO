import os
import numpy as np
import pandas as pd
from random import randint
import pickle
import time
import lib.fcns_asstopt as fcns_asstopt
#from lib.utilities import revenue_MM
import sys


def sigma_digest_save_result(sigmas, lambdas, nb_prod, d):
    sigma_res= []
    
    a = Sigma[np.nonzero(Lambda)]
    b = Lambda[np.nonzero(Lambda)]
    sigmas_sorted = a[np.argsort(b), :][::-1]
    lambdas_sorted = b[np.argsort(b)][::-1]
    for i in range(len(sigmas_sorted)):
        nb_prod_indiff = (sigmas_sorted[i, :] == nb_prod - 1).sum()
        prod_prefered = np.argsort(sigmas_sorted[i, :])  # including the indiff products
        #sigma_res.append((i,lambdas_sorted[i]))    
        # print the list of preferred in order of preference
        
        res = []
        dd = prod_prefered[:nb_prod - nb_prod_indiff]
    
        for el in dd:
            for p, v in enumerate(d):
                if( el == p and p >0 and el > 0):
                    res.append(v[1])
        #print(res)
        sigma_res.append( (i,lambdas_sorted[i], res) )
               
    return sigma_res


algo_chosen = 'GDT'
data_version =  '030' 
nor = '3' # number of runs

filename_transaction        = 'transaction_data_bal_'+  data_version + '.dat'
filename_choice_model_GDT   = 'choice_model_GDT_bal_'+  data_version + '.dat'
filename_choice_model_BM    = 'choice_model_BM_' +  data_version + '.dat'
    
filename_aso  = 'aso_bal_'+  data_version + '.dat'
filename_output  = 'output_bal_'+  data_version + '.dat'
    
script_dir  = '/home/oleh/assortment_optimization-master/sample'
    #os.path.dirname(__file__) 
    #'/home/oleh/assortment_optimization-master/sample'
rel_path_transaction = "data/"+filename_transaction
abs_file_transaction = os.path.join(script_dir, rel_path_transaction)
    
rel_path_choice_model_GDT = "data/"+filename_choice_model_GDT
abs_file_choice_model_GDT = os.path.join(script_dir, rel_path_choice_model_GDT)
    
rel_path_choice_model_BM = "data/"+filename_choice_model_BM
abs_file_choice_model_BM = os.path.join(script_dir, rel_path_choice_model_BM)

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
    
with open(abs_file_choice_model_GDT, 'rb') as sales:
    my_depickler = pickle.Unpickler(sales)
    sigma_GDT_sorted =   my_depickler.load()
    lambda_GDT_sorted =   my_depickler.load()    
    
(nb_asst, nb_prod) = Inventories.shape
Lambda = lambda_GDT_sorted[:10] #nb of lambds that explains most of the sales !!!! 
Sigma = sigma_GDT_sorted[:10]   

d = []   
for el in dic:
    d.append(el)   
    
sigma_result = sigma_digest_save_result (Sigma,Lambda, nb_prod,d)    

#get sigmas in right format
sigma_prod_list = []
for item in sigma_result:
    sigma_prod_list.append((item[1], item[2]))

#get Rev_Baseline
Rev_BaselineNew = 0 
for product in real_revenue:
    for row in sigma_prod_list:
        if (product[0] in row[1]):
            #if (len(row[1]) < 2):
            Rev_BaselineNew +=product[1]*row[0]


input_data =[5,10,15,20,25,30,35,40,45]
main_result = []


for min_num in input_data:
    print(min_num)
    min_capacity = min_num # len(Prod_List_Max)
    max_capacity = 1000#unconstrained

    verbose = True #should GUROBI print the steps of solving of the problem?

    t1 = time.time()
    [x_found, obj_val] = fcns_asstopt.run_asstopt_GDT(Lambda, Sigma, Revenue[:len(Sigma.T)], min_capacity, max_capacity,
                                                      verbose=verbose)
    t2 = time.time()
   
    aso_result = []
    
    for el, val in enumerate(x_found):
        for i, v in enumerate(d):
            if(val == True and el == i and i >0 and el > 0):
                aso_result.append(v[1])
    
    same_prod= np.intersect1d(Prod_List_Max,  aso_result)
    diff_prod = np.setdiff1d(aso_result, Prod_List_Max)   
    inc_rev = (obj_val - Rev_BaselineNew ) / Rev_BaselineNew
   
    main_result.append( (min_num, t2-t1, len(aso_result),aso_result,list(same_prod),list(diff_prod), obj_val, Rev_BaselineNew,inc_rev ) )

    print((min_num, t2-t1, len(aso_result),aso_result,list(same_prod),list(diff_prod), obj_val, Rev_BaselineNew,inc_rev ))


'''   
filename_result_       = 'result_with_prod_bucket_data_bal_'+ nor + '_' + data_version + '.dat'
script_dir  = '/home/oleh/assortment_optimization-master/sample'
    #os.path.dirname(__file__) 
    #'/home/oleh/assortment_optimization-master/sample'

abs_filename_result_  = os.path.join(script_dir, filename_result_)

with open(filename_result_ ,'wb') as aso_:
    my_pickler = pickle.Pickler(aso_,protocol=2)
    my_pickler.dump(main_result)




with open(abs_filename_result_, 'rb') as aso__:
    my_depickler = pickle.Unpickler(aso__)
    main_result =     my_depickler.load()
''' 
import matplotlib.pyplot as plt
for  rev_history in  main_result:
    #if (  (len(rev_history[5]) > 0) and  (rev_history[0] not in (21,22,23,24)) ):  
    #if ( len(rev_history[5]) > 0):  
    plt.scatter( x = rev_history[0], y =  rev_history[-1] , c = 'red')     
plt.xlabel("Number of products")
plt.ylabel("Increase of revenue vs baseline")   
#plt.legend()
plt.show()   
    
for  rev_history in  main_result:
    #if (  (len(rev_history[5]) > 0) and  (rev_history[0] not in (21,22,23,24)) ):  
    if ( len(rev_history[5]) > 0):  
        plt.scatter( x = rev_history[0], y =  len(rev_history[5]), label = len(rev_history[5]) )  
plt.xlabel("Number of products")
plt.ylabel("Number of new recommended products")   
plt.legend()
plt.show() 
    