import os
import numpy as np
import pandas as pd
from random import randint
import pickle
import time
import lib.fcns_asstopt as fcns_asstopt
#from lib.utilities import revenue_MM
import sys

print("Optimization of the assortment given a choice model")

min_capacity = 5# len(Prod_List_Max)
max_capacity = 2000#unconstrained

verbose=True #should GUROBI print the steps of solving of the problem?

algo_chosen = 'GDT'
data_version =  '030'   
    
#In the case of algo_chosen, those parameters tune the number of sub-columns to boost the generation
threshold = 0.01
min_sub_col_per_col = 1

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
 
print('')    
print('The assortment with higher real revenue has been found ! Number of products:' , len(Prod_List_Max))
if (min_capacity < len(Prod_List_Max)):
    print('The min capacity constraint has been violated ! It should be ',len(Prod_List_Max) , 'at least.')
    #sys.exit()
print('')  
 
if(algo_chosen=='GDT'):    
    print("Opening choice model, format GDT")
    with open(abs_file_choice_model_GDT, 'rb') as sales:
        my_depickler = pickle.Unpickler(sales)
        sigma_GDT_sorted =   my_depickler.load()
        lambda_GDT_sorted =   my_depickler.load()    
elif(algo_chosen=='BM'):
    print("Opening choice model, format BM")
    with open(abs_file_choice_model_BM, 'rb') as sales:
        my_depickler = pickle.Unpickler(sales)
        sigma_BM_sorted =   my_depickler.load()
        lambda_BM_sorted =   my_depickler.load()
else:
    print("Error; wrong input parameter, which algorithm do you wish to use?")
    
(nb_asst, nb_prod) = Inventories.shape   

'''
adj_lambdas_num = 10 
Lambda = lambda_GDT_sorted[:adj_lambdas_num]  
Sigma = sigma_GDT_sorted[:adj_lambdas_num]   
'''

if(algo_chosen=='GDT' or algo_chosen=='gen' ):
    #adj_lambdas_num = 10 #----> number of lambdas that explains more of the sales
    Lambda = lambda_GDT_sorted #[:adj_lambdas_num]  
    Sigma = sigma_GDT_sorted   #[:adj_lambdas_num]    
    t1 = time.time()
    #fcns_asstopt.run_asstopt_GDT
    [x_found, obj_val] = fcns_asstopt.run_asstopt_GDT(Lambda, Sigma, Revenue[:len(Sigma.T)], min_capacity, max_capacity,
                                                      verbose=verbose)
    t2 = time.time()
    
elif(algo_chosen=='BM'):
    Lambda = lambda_BM_sorted
    Sigma = sigma_BM_sorted
    t1 = time.time()
    [x_found, obj_val] = fcns_asstopt.run_asstopt(Lambda, Sigma, Revenue[:len(Sigma.T)], min_capacity, max_capacity,
                                                      verbose=verbose)
    t2 = time.time()
else:
    print("Error; wrong input parameter, which algorithm do you wish to use?")
    
    

print("Saving Optimal assortment ")
with open(abs_file_aso, 'wb') as aso:
    my_pickler = pickle.Pickler(aso,protocol=2)
    my_pickler.dump(x_found)
    my_pickler.dump(obj_val)
    my_pickler.dump(t2-t1)
    
d = []   
for el in dic:
    d.append(el)    

"""
with open(abs_file_aso, 'rb') as aso:
    my_depickler = pickle.Unpickler(aso)
    x_found =     my_depickler.load()
    obj_val =     my_depickler.load()
    Tp =         my_depickler.load()

""" 

aso_result = []
    
for el, val in enumerate(x_found):
    for i, v in enumerate(d):
        if(val == True and el == i and i >0 and el > 0):
            aso_result.append(v[1])
    
same_prod= np.intersect1d(Prod_List_Max,  aso_result)
diff_prod = np.setdiff1d(aso_result, Prod_List_Max)    
"""
with open(abs_file_output, 'rb') as out:
    my_depickler = pickle.Unpickler(out)
    Res_ass =    my_depickler.load()
    Res_sigma =     my_depickler.load()
"""     




def sigma_digest(sigmas, lambdas, nb_prod):
    a = sigmas[np.nonzero(lambdas)]
    b = lambdas[np.nonzero(lambdas)]
    sigmas_sorted = a[np.argsort(b), :][::-1]
    lambdas_sorted = b[np.argsort(b)][::-1]
    for i in range(len(sigmas_sorted)):
        nb_prod_indiff = (sigmas_sorted[i, :] == nb_prod - 1).sum()
        prod_prefered = np.argsort(sigmas_sorted[i, :])  # including the indiff products
        print("Sigma number ", i, ", probability associated:", lambdas_sorted[i], ", prefered products in order:")
        # print the list of preferred in order of preference
        print(prod_prefered[:nb_prod - nb_prod_indiff])
    return 1

#sigma_digest(Sigma,Lambda, nb_prod)



def sigma_digest2(sigmas, lambdas, nb_prod):
    a = sigmas[np.nonzero(lambdas)]
    b = lambdas[np.nonzero(lambdas)]
    sigmas_sorted = a[np.argsort(b), :][::-1]
    lambdas_sorted = b[np.argsort(b)][::-1]
    for i in range(len(sigmas_sorted)):
        nb_prod_indiff = (sigmas_sorted[i, :] == nb_prod - 1).sum()
        prod_prefered = np.argsort(sigmas_sorted[i, :])  # including the indiff products
        print("Sigma number ", i, ", probability associated:", lambdas_sorted[i], ", prefered products in order:")
        # print the list of preferred in order of preference
        
        res = []
        dd = prod_prefered[:nb_prod - nb_prod_indiff]
    
        for el in dd:
            for p, v in enumerate(d):
                if( el == p and p >0 and el > 0):
                    res.append(v[1])
        print(res)
               
    return 1

#sigma_digest2(Sigma,Lambda, nb_prod)



def sigma_digest_save_result(sigmas, lambdas, nb_prod):
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



sigma_result = sigma_digest_save_result (Sigma,Lambda, nb_prod)

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


#test    
#Rev_BaselineNew += ggg                
"""           
test = []   
test2 = []       
Rev_BaselineNew = 0 
for product in real_revenue:
    for row in sigma_prod_list:
        if (product[0] in row[1] and product[0] not in test):
            #if (len(row[1]) < 2):
            #Rev_BaselineNew +=product[1]*row[0]
            #test.append(product[0] )
            #print(product[0] , product[1], row[0])
            test2.append((product[0] , product[1], row[0]))
            
dff = pd.DataFrame(test2 )            
rr =  dff.groupby(dff[0]).mean() 
rr['pred'] = rr[1]*rr[2]   
ggg =  rr['pred'].values.sum()         
"""            
                
                
#all predicted by GDT model revenues   
pred_data_res = []                
for k,l in enumerate(ass_data):
    for i,j in enumerate(products_data):  
        for sigma in  sigma_prod_list:
            if ((l,j) in predicted_rev_data.index):
                if(j in sigma[1]):
                    if (len(sigma[1]) < 2):
                        pred_data_res.append( (l, predicted_rev_data[(l,j)]*sigma[0] ))
          
df = pd.DataFrame(pred_data_res)    
predicted_max_revenue_= np.float64(df.groupby(df[0]).sum().max())                    



print("Saving results")
with open(abs_file_output, 'wb') as out:
    my_pickler = pickle.Pickler(out,protocol=2)
    my_pickler.dump(aso_result)
    my_pickler.dump(sigma_result)



print("Generated data, in file ", filename_transaction, "with nb_prod =", nb_prod, "products")
print('')
print(nb_asst, "assortments available in transaction dataset")
print('')
print("Optimal assortment found in", t2-t1, "seconds. Number of products", len(aso_result), ".Products present in optimal assortment:")
print('')
print(aso_result) #x_found
print('')
print("List of common products in assorment with higher real revenue and optimal assortment: ")
print('')
print(list(same_prod))
print('')
print("List of new recommended products: ")
print('')
print(list(diff_prod))
print('')
print("List of excluded products: ")
print('')
print (list(np.setdiff1d(Prod_List_Max , same_prod)))
print('')
print("Expected revenue of the optimal assortment:")
print('{:04.2f}'.format(obj_val))
print("Expected baseline revenue according to GDT model:")
print('{:04.2f}'.format(Rev_BaselineNew) , " with max-method (",predicted_max_revenue_,")" )
print("Increase of revenue vs baseline:")
print('{:04.2%}'.format( (obj_val - Rev_BaselineNew ) / Rev_BaselineNew))
print('')
print("Assortment optimization completed")

#sigma_digest2(Sigma,Lambda, nb_prod)
    
#(nb_col, nb_prod) = Sigma.shape
'''
y = []  
for k in range(nb_col):
    for i in range(nb_prod):
        y.append( (  (k, i),Lambda[k], Revenue[i] , Lambda[k] * Revenue[i] ))
    
'''
