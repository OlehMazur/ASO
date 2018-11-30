#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:16:49 2018

@author: oleh
"""

import dask.dataframe as dd 
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import Imputer
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import datetime
#from sklearn.cluster import AgglomerativeClustering
#import scipy.cluster.hierarchy as sch

time1 = time.time()

def get_week(year, month , day):
    dt = datetime.date(year, month, day)
    wk = dt.isocalendar()[1]
    return wk

df_2018 = dd.read_csv("BG_2018_v2.csv", encoding = 'cp1251', delimiter = ";", decimal = "," 
                      ,dtype = {'Количество продукции Шт': 'float64'}
                      )
df_2018.columns = ['Shop_Id', 'SKU_Id', 'Date', 'Quantity', 'SalesValueDal', 'SalesValue']

#sample_2017 = df_2017.head(1000)

'''
sku_dic = dd.read_csv("SKU.csv",  encoding = 'cp1251', delimiter = "|" , engine = 'python',  error_bad_lines=False)
sku_dic.columns = ['SKU_Id', 'Client', 'grpName', 'xcode', 'xname', 'Category', 'Producer', 'SKU_Nielsen_Id', 'DateCreated']
'''
sku_dic = pd.read_csv("BG_V2_SKU.csv",  encoding = 'cp1251', delimiter = ";" , engine = 'python',  error_bad_lines=False)
sku_dic_data = sku_dic.iloc[:, [0,1,2,3,4,5,6]]
sku_dic_data.columns = ['SKU_Id', 'Client', 'xcode', 'xname', 'Category', 'Producer', 'SKU_Nielsen_Id']

shop_dic = pd.read_csv("BG_V2_Shop.csv",encoding = 'cp1251', delimiter = ";" , engine = 'python',  error_bad_lines=False)
shop_dic_data = shop_dic.iloc[:, [0,1,2,4,6,8,9,10,11,12,13]]
shop_dic_data.columns = ['Shop_Id', 'Network', 'Store_nb', 'Format', 
                         'UnionNetwork', 'SubChannel', 'Region', 'Province', 'City', 'Population', 'UnionRegion' ]

'''
shop_dic = dd.read_csv("Shop.csv",  encoding = 'cp1251', delimiter = "|" , engine = 'python',  error_bad_lines=False)
shop_dic.columns = ['Shop_Id', 'Network', 'Store_nb', 'Store_Add', 'Format', 'UnionNetwork', 'SubChannel']
#s = shop_dic.compute()
'''

'''
geo_dic = dd.read_csv("Geo.csv",  encoding = 'cp1251', delimiter = "|" , engine = 'python',  error_bad_lines=False)
geo_dic.columns = ['Geo_Id', 'Region', 'Province', 'City', 'Population', 'UnionRegion', 'GIBDD_Region', 'UnionSBP', 'Store_Add']
#g = geo_dic.compute()
'''

ff  = dd.merge(df_2018[["Shop_Id", "SKU_Id", "Date", "Quantity", "SalesValue"]],
         shop_dic_data[["Shop_Id", "Network", "City"]],
         how = 'left',
         on = 'Shop_Id')

ff2 = dd.merge(ff, 
               sku_dic_data[["SKU_Id","Producer", "Category", "xname"]], 
               how = 'left',
               on = "SKU_Id"               
               )
'''
#should be the Geo_Id in FACT table !!!!!
ff3 = dd.merge(ff2,
               geo_dic[["Store_Add", "City"]],
               how = 'left',
               on = "Store_Add"
               )
'''               
'''(ff["Client"] == 'Пятёрочка') & МАГНИТ''' 

ff2 = ff2[ (ff2["Network"] == 'Пятёрочка') & (ff2["Category"] == 'Пиво' ) & (ff2["Producer"] == 'BALTIKA') & (ff2["City"] == 'Москва' )]
#t = ff2.compute()

tt = ff2.groupby(['Shop_Id']).agg({'Quantity': np.sum, 'SalesValue': np.sum})
tt["Assortment"] = ff2.groupby(['Shop_Id'])['SKU_Id'].nunique()

result = tt.compute()

#sku_dic.groupby(sku_dic["Client"]).size().compute()

# make a cluster
X = result.iloc[:, [1,2]] # SalesValue and Assortments
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

wcss =[]
for i in range(1,11):
    kmeans = KMeans(
            n_clusters=i, 
            init = 'k-means++', 
            max_iter = 300, 
            n_init= 10, 
            random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('Elbow Method', dpi = 300)
plt.show()

kmeans = KMeans(
        n_clusters=4, 
        init = 'k-means++', 
        max_iter = 300,
        n_init = 10,
        random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans ==0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans ==1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans ==2, 0], X[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans ==3, 0], X[y_kmeans == 3, 1], s = 50, c = 'magenta', label = 'Cluster 4')
#plt.scatter(X[y_kmeans ==4, 0], X[y_kmeans == 4, 1], s = 50, c = 'yellow', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 100, c = 'gray', label = 'Centroids')
#plt.scatter(x = bal_prod[:,1], y=bal_prod[:,2] , s = 100, c = '', edgecolors = 'black', marker= 'o',alpha = 1.0, label = 'Baltika product')
plt.title('Clustering the stores (k-means++)')  
plt.xlabel('Sales Value')   
plt.ylabel('Assortment)')  
plt.legend()
plt.show()    


#data with cluster
result["Cluster"] = y_kmeans

#result_cl3 = result.loc[result["Cluster"] == 3] #has fairly high sales
#data_cl2 = result.loc[result["Cluster"] == 2] 
data_cl1 = result.loc[result["Cluster"] == 1]
data_cl1["Shop_Id"] = data_cl1.index
#data_cl0 = result.loc[result["Cluster"] == 0]

#input file preparation 
data_with_shop_cluster = dd.merge(ff2, 
         data_cl1[['Shop_Id']],
         how = 'inner',
         on = 'Shop_Id'
        )


data_with_shop_cluster_df= data_with_shop_cluster.compute()
data_with_shop_cluster_df["year"]=  (data_with_shop_cluster_df["Date"].astype(int)/10000).astype(int)

data_with_shop_cluster_df["month"]= (data_with_shop_cluster_df["Date"].astype(int)/100).astype(int) - \
                                    (data_with_shop_cluster_df["Date"].astype(int)/10000).astype(int) *100

data_with_shop_cluster_df["day"] = (data_with_shop_cluster_df["Date"].astype(int)).astype(int)  - \
                                    (data_with_shop_cluster_df["Date"].astype(int)/100).astype(int) *100

     
data_with_shop_cluster_df["week"]  = pd.to_datetime(data_with_shop_cluster_df[["year", "month", "day"]]).dt.week                      
#get_week(2018,11,28)

data_with_shop_cluster_df["SalesNumber"] = data_with_shop_cluster_df["SKU_Id"]

main = data_with_shop_cluster_df.groupby(['week', 'Shop_Id', 'SKU_Id', 'xname'], as_index = False) \
                .agg({'Quantity': np.sum, 'SalesValue': np.sum, 'SalesNumber': np.size})
             
main["AssKey"] = main["week"].astype(str) + main["Shop_Id"].astype(str)



tns = data_with_shop_cluster_df.groupby(['week', 'Shop_Id'], as_index = False) \
                .agg({'Quantity': [np.min, np.max, np.sum]})
                
tns["TotalNumOfSales"] = data_with_shop_cluster_df.groupby(['week', 'Shop_Id'])['SKU_Id'].nunique().reset_index(drop=True)
tns.columns = ['week', 'Shop_Id', 'minQuantity', 'maxQuantity', 'TotalSalesPerAss' , 'TotalNumOfSales']
tns["AssKey"] = tns["week"].astype(str) + tns["Shop_Id"].astype(str)


rev = data_with_shop_cluster_df.groupby(['SKU_Id'], as_index = False) \
                .agg({'SalesValue': np.mean})
rev.columns = ['SKU_Id', 'Revenue']


output_data_p1 = pd.merge (
        main[['AssKey', 'week','Shop_Id', 'SKU_Id', 'xname', 'Quantity', 'SalesValue', 'SalesNumber'] ], 
        tns[['AssKey', 'minQuantity', 'maxQuantity', 'TotalSalesPerAss','TotalNumOfSales']],
        how='left',
        on= 'AssKey')

output_data_p2 = pd.merge (
        output_data_p1,
        rev,
        how= 'left',
        on = 'SKU_Id'
        )

output_data_p2["No-purchaseProb"] = 0.1+0.2*( output_data_p2["TotalNumOfSales"] / (output_data_p2["maxQuantity"] - output_data_p2["minQuantity"] ) )
output_data_p2["PurchaseProb"] = (1 - output_data_p2["No-purchaseProb"] ) * (output_data_p2["Quantity"] / output_data_p2["TotalSalesPerAss"] ) 
output_data_p2 = output_data_p2[["AssKey", "SKU_Id", "xname", "Revenue", "No-purchaseProb", "PurchaseProb"]]

#encoding = 'cp1251'
output_data_p2.to_csv("GDT_Input.csv", sep = '|', index = False )

time2 = time.time()
print('done !')
print(time2-time1, 'sec')







