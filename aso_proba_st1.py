#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:04:11 2018

@author: oleh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

print("################################# File generation_data_products_based.py #################################")
print("Generation of the training and test sets")
print("Use of the MMNL choice model to generate utilities on products.")

data_version = '030'
dataset = pd.read_csv("GDT_20.csv", delimiter = ";", decimal = ",", index_col = [0,1])

Proba_product_ = dataset.iloc[:, -1]
Proba_product_nc = dataset.iloc[:, -2]
Revenue = dataset.iloc[:, 0]

products = []
ass = []
products_data = dataset.index
for pr in products_data:
    products.append(pr[1])
    ass.append(pr[0])

products = np.unique(products)
products = np.insert(products, 0,0) #no choice
ass = np.unique(ass)

nb_prod = len(products) 
nb_ass =  len(ass) # + no choice

input_data = np.zeros((nb_ass, nb_prod), dtype = np.float64)
revenue = np.zeros(nb_prod, dtype = np.float64)
inventories = np.zeros_like(input_data, dtype=bool)  
dic= np.zeros(nb_prod,dtype=object)
ass_revenue = np.zeros(nb_ass, dtype = np.float64)
rev_prod_list = []
#rev_predicted = []


ass_revenue = []

for k,l in enumerate(ass):
    for i,j in enumerate(products):          
        if ((l,j) in Proba_product_nc.index):
            input_data[k,0] = Proba_product_nc[(l,j)]
            revenue[0] = 0
            inventories[k,0]  = True
            
            #if((l,j) == ('138308', 57275)):
                #print (input_data[k,0])
                
        if ((l,j) in Proba_product_.index):
            input_data[k,i] = Proba_product_[(l,j)]
            revenue[i] = Revenue[((l,j))]
            inventories[k,i] = True
            dic[i] =  (i, j)           
            #ass_revenue.append((l, Proba_product_[(l,j)] *Revenue[(l,j)]))
            ass_revenue.append((l, Revenue[(l,j)]))
            rev_prod_list.append ( (j,Revenue[(l,j)] ))   
            #rev_predicted.append((l,j,Revenue[(l,j)] ))
            #if((l,j) == ('138308', 57275)):
                #print (input_data[k,i])
                #print(revenue[i])



df = pd.DataFrame(ass_revenue)    
ass_revenue_res = np.float64(df.groupby(df[0]).sum().max())  
ss = df.groupby(df[0]).sum()
max_ass_num = np.array(ss.sort_values(ss.columns[0], ascending = False).index)[0]
prod_list_data = np.array(products_data)
prod_list_max = []
for z in prod_list_data:
    if (z[0] == max_ass_num ):
        prod_list_max.append(z[1])

#get tuples(prod, revenue) for assortment with max real revenue        
rev_max_prod = []       
for el in  prod_list_max:
    for el2 in rev_prod_list:
        if (el == el2[0]):
            rev_max_prod.append((el, el2[1]))
            
               
rev_max_prod  = list(set(rev_max_prod))
       
        
filename_transaction = 'transaction_data_bal_'+data_version+'.dat'

#ensures that the path is correct for data file
script_dir = '/home/oleh/assortment_optimization-master/sample' #absolute dir the script is in
rel_path_transaction = "data/"+filename_transaction
abs_file_transaction = os.path.join(script_dir, rel_path_transaction)

#we save the results into a file
#Use of protocol 2 to ensure back-compatibility with Python 2.7
with open(abs_file_transaction, 'wb') as sales:
    my_pickler = pickle.Pickler(sales,protocol=2)
    my_pickler.dump(inventories)
    my_pickler.dump(input_data)
    my_pickler.dump(revenue)
    my_pickler.dump(dic)
    my_pickler.dump(ass_revenue_res)
    my_pickler.dump(prod_list_max)
    my_pickler.dump(rev_max_prod )
    my_pickler.dump(Revenue) #test
    my_pickler.dump(products) #test
    my_pickler.dump(ass) #test


    
print("End of generation of data. File ", filename_transaction, "has been saved in /sample/data/.")
print("#######################################################################################")
#  end of exportation of the generated data
#####################################

print("Average probability of no-choice:", np.average(input_data[:,0]))



                