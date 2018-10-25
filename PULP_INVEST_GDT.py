import numpy as np
import time
import lib.utilities as utilities
import lib.gen_GDT_PULP as gen_GDT
import lib.pulp_GDT as pGDT
import pickle
import os
import numpy as np
import pickle
import sys
from pulp import *
from gurobipy import *



NB_ITER =5 #20  #if eps_stop is different than 0, NB_ITER ignored
eps_stop= 0
data_version = '096'
algo_chosen  = 'GDT'  #'GDT'

#############################
#  run_GDT runs the GDT algorithm, generating columns at each iteration and optimizing with repeated calls to the master problem
#  inputs: Inventories, sales data Proba_product, stop criterion ITERATIONS_MAX and eps_stop
#  returns a GDT choice model (sigma_GDT_sorted, lambda_GDT_sorted), as well as the history of reduced costs to track the learning efficiency

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



t1 = time.time()
 # definition of the parameters according to the data
(nb_asst, nb_prod) = Inventories.shape
    # Inventories=Inventories.astype(bool)#to use the ivt as selectors
v = Proba_product.T  # to be consistent with the notations of the article of BM
history_obj_val = np.zeros(1, dtype=np.float32)


# Initialization: we built nb_prod possible columns of A, as specified in the thesis
# nb_col = nb_prod#for the initialization
sigma_GDT = np.full((nb_prod, nb_prod), fill_value=nb_prod - 1, dtype=np.int32)
A = np.full((nb_prod, nb_prod, nb_asst), fill_value=0,
                dtype=np.float64)  # first component is the number of columns generated at the begining
for k in range(nb_prod):
    sigma_GDT[k, k] = 0
    A[k, :, :] = sigma2a(sigma_GDT[k, :], Inventories)

(nb_col,nb_prod,nb_asst) = A.shape

'''
[lambda_found, alpha_found, nu_found, obj_val_master] = \
        pGDT.restricted_master_main_PULP(A, v, prob, verbose=False)
history_obj_val[0] = obj_val_master
'''

prob = pulp.LpProblem("finding_lambda", pulp.LpMinimize)
    
lmbda = {}

# Create variables
for k in range(nb_col):
    lmbda[k] = pulp.LpVariable('lambda_%s' % k,  cat='Continuous', lowBound= 0)
if nb_col==0:
    lmbda[0] = pulp.LpVariable('lambda_0',  cat='Continuous', lowBound= 0)

eps_p = {}
eps_m = {}
       
for i in range(nb_prod):
    for m in range(nb_asst):
        eps_p[i,m] = pulp.LpVariable('eps_p_%s_%s' % (i,m), lowBound= 0)           
        eps_m[i,m] = pulp.LpVariable('eps_m_%s_%s' % (i,m), lowBound= 0)       

 
prob+=lpSum([eps_p, eps_m ])   

#Create constraints
var = [lmbda[k] for k in range(nb_col)]

for i in range(nb_prod):
    for m in range(nb_asst):                 
        prob+= lpSum( A[:,i,m] * var)  + eps_p[i,m]- eps_m[i,m] - v[i,m] == 0 , 'distance_%s_%s' % (i,m)
       
if nb_col > 0:
    prob+= lpSum( var) == 1 , 'sum_to_%s' %1
      
#prob.writeLP("RMP_PULP.lp")
prob.solve()
print(value(prob.objective))
print("Status:", LpStatus[prob.status])

#definition of the return variables with expected shape
return_lmbda_pulp = np.zeros(max(nb_col,1))
alpha_pulp = np.zeros((nb_prod, nb_asst))
nu_pulp = np.zeros(1)
    
# Extraction of the solutions
obj_value = value(prob.objective)
    
  
if nb_col > 0:
    for k in range(nb_col):
        return_lmbda_pulp[k] = lmbda[k].varValue 
else:
    return_lmbda_pulp[0] = 0
    
#Extraction of the dual values at optimality

constraints_pulp = prob.constraints
 #the constraint sum_to_1 is the last one recorded; we take its dual value
nu_pulp[0] = constraints_pulp[list(constraints_pulp)[-1]].pi    
    
#We take the dual value of the constraints distance_i_m
constr_list_pulp = list(constraints_pulp)
for i in range(nb_prod):
    for m in range(nb_asst):
        alpha_pulp[i,m] = constraints_pulp[constr_list_pulp[i*nb_asst+m]].pi

if verbose:
    prob.writeLP("RMP.lp")
 

#return([repair_lambda(return_lmbda), alpha, nu, obj_value])   



 #Creating the Model using Gurobi
verbose = False
model = Model('finding_lambda')
model.setParam( 'OutputFlag', verbose )

'''choose the method:
    Primal simplex(method=0)
    Dual simplex(method=1)
    Barrier(method=2)
    Non - deterministic Concurrent(method=3)'''
    
method = 2
model.setParam("Method", method);

# Create variables
lmbda = {}
for k in range(nb_col):
    lmbda[k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='lambda_%s' % k,obj=0)
if nb_col==0:
    lmbda[0] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='lambda_0',obj=0)
eps_p = {}
eps_m = {}
    
for i in range(nb_prod):
    for m in range(nb_asst):
        eps_p[i,m] = model.addVar(lb=0, name='eps_p_%s_%s' % (i,m), obj=1)
        eps_m[i,m] = model.addVar(lb=0, name='eps_m_%s_%s' % (i,m), obj=1)
model.ModelSense = 1 #Minimization
model.update()
#Create constraints
var = [lmbda[k] for k in range(nb_col)]
[[model.addConstr(LinExpr(A[:,i,m], var)+ eps_p[i,m] - eps_m[i,m] - v[i,m] == 0, 'distance_%s_%s' % (i,m) ) for m in range(nb_asst)] for i in range(nb_prod)]
if nb_col > 0:
    model.addConstr( LinExpr([1 for i in range(nb_col)], var) == 1, name='sum_to_%s' %1)

model.write("RMP.lp")
model.optimize()
obj_value = model.ObjVal
print(obj_value)
   
return_lmbda = np.zeros(max(nb_col,1))
alpha = np.zeros((nb_prod, nb_asst))
nu = np.zeros(1)
   
if nb_col > 0:
    for k in range(nb_col):
        return_lmbda[k] = lmbda[k].X
else:
    return_lmbda[0] = 0

 #Extraction of the dual values at optimality
constraints = model.getConstrs() 
    #the constraint sum_to_1 is the last one recorded; we take its dual value
nu[0] = constraints[-1].getAttr('Pi') 

 #We take the dual value of the constraints distance_i_m
for i in range(nb_prod):
    for m in range(nb_asst):
        alpha[i,m] = constraints[i*nb_asst+m].getAttr('Pi') 
        
if verbose:
    model.write("RMP.lp")
    #model.write("RMP.sol")

#return([repair_lambda(return_lmbda), alpha, nu, obj_value])     


def repair_lambda(lambda_found):
    lambda_found[lambda_found<0]=0
    return lambda_found/lambda_found.sum()   

#############################
#  Various functions
#
#
# User-friendly printing of the choice model  
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


# Computes the reduced cost of all columns of A, given rc=-alpha * a - nu
def reduced_cost_matrix(A, alpha, nu):
    return -np.sum(A * alpha, axis=(1, 2)) - nu


# returns the n_new_branches smallest reduced costs (and their sigma, A associated), taken from all the possible k defined by set_k_possible
def lowest_reduced_cost(set_k_possible, sigma_GDT, nb_prod, alpha_found, nu_found, assortments, n_new_branches=100):
    nb_col = len(sigma_GDT)
    new_sigma_GDT = sigma_GDT
    for k in set_k_possible:
        new_new_sigma_GDT = add_new_sigma_GDT(sigma_GDT[k, :], nb_prod)
        new_sigma_GDT = np.concatenate((new_sigma_GDT, new_new_sigma_GDT), axis=0)
    new_sigma_GDT = new_sigma_GDT[nb_col:, :]  # we exclude the columns already in the dictionnary
    new_A = multiple_sigma2a(new_sigma_GDT, assortments)
    new_rc = reduced_cost_matrix(new_A, alpha_found, nu_found)
    sort = np.argsort(new_rc)[:min(n_new_branches, len(
        new_rc))]  # we take the n_new_branches smallest rc (exception if n_new_branches is > len(rc) )
    return [new_sigma_GDT[sort, :], new_A[sort, :, :], new_rc[sort]]


# heuristically choose a component of lambda_found (and returns the indicium associated) with a softmax
def choose_n(lambda_found, n):
    # return np.random.choice(len(lambda_found), p=softmax(lambda_found))
    # returns n examples different; if n>nb of nonzero components of lambda_found, we return this number of numbers.
    
    return np.random.choice(len(lambda_found), size=min(n, len(np.nonzero(lambda_found)[0])), replace=False,
                            p=lambda_found)
  

# computes the reduced cost of a column expressed as sigma
def reduced_cost(sigma, alpha, nu, Inventories):
    return - np.sum(np.sum((alpha * sigma2a(sigma, Inventories)), axis=0), axis=0) - nu


# returns the product chosen by the customer defined by sigma when the assortment asst is displayed to him
def product_chosen(sigma, asst):
    if np.sum(asst, axis=0) == 0:
        return -1
    # value is the rank of preference in sigma of the prefered product present in the assortment
    value = np.min(sigma[np.nonzero(asst)])
    where_is_value = np.where(sigma == value)[0]
    if (len(where_is_value) == 1):
        return where_is_value[0]
    elif (len(where_is_value) == 0 or len(where_is_value) >= 2):
        return -1


# Returns the column a associated to a sigma and several assortments
# the last block is sufficient, but we can speed up the algo (approx *3) with the first three blocks that deal with the singularities very efficiently (cases where the 1st, 2nd or no product is chosen)
def sigma2a(sigma, assortments):
    nb_asst = len(assortments)
    nb_prod = len(sigma)
    ret = np.zeros((nb_prod, nb_asst))

    # three blocks to improve efficiency:
    # we have to process the assts_not_yet_processed, that will store the assortments not catched by the speed-ups
    # 1. when the preferred product of sigma is present in the assortments, we fill the assortments
    preferred_prod = np.where(sigma == 0)[0][0]
    ret[preferred_prod, assortments[:, preferred_prod]] = 1
    assts_not_yet_processed = np.arange(nb_asst)[np.invert(assortments[:, preferred_prod])]

    # 2. same thing for the second product; we may have only the first product ranked in sigma, that is why we put a 'try' block
    try:
        scd_preferred_prod = np.where(sigma == 1)[0][0]
        ret[scd_preferred_prod, np.invert(assortments[:, preferred_prod]) & assortments[:, scd_preferred_prod]] = 1
        assts_not_yet_processed = np.arange(nb_asst)[np.invert(assortments[:, preferred_prod]) & np.invert(
            assortments[:,
            scd_preferred_prod])]  # absence of preferred product and presence of second preferred product
    except:
        a = 1  # useless

    # 3. catching the assortments in which we have none of the products ordered
    # it is equivalent to 'the products ranked==nb_prod-1 (ie not ordered) are present in number equal to the size of the assortment
    asst_without_choice = \
    np.where(assortments[:, np.where(sigma == nb_prod - 1)[0]].sum(axis=1) == assortments.sum(axis=1))[
        0]  # True when  the two previous numbers are equal: ie we know that no product preferred is in the asst!
    nb_prod_pst_in_asst = assortments.sum(axis=1)
    for m in asst_without_choice:
        ret[assortments[m, :], m] = 1 / nb_prod_pst_in_asst[m]

    assts_not_yet_processed = np.setdiff1d(assts_not_yet_processed, asst_without_choice)

    for m in assts_not_yet_processed:
        prod_chosen = int(product_chosen(sigma, assortments[m][:]))  # is equal to -1 if no product chosen
        if (prod_chosen != -1):
            ret[prod_chosen, m] = 1
        # module added to put 1/nb_prod_pst_in_asst at the components if no product chosen: GDT
        else:
            nb_prod_pst_in_asst = assortments[m, :].sum()
            ret[assortments[m, :].astype(bool), m] = 1 / nb_prod_pst_in_asst
    return ret


# get the matrix A corresponding to a list of columns sigmas
# calls the function sigma2a several times
def multiple_sigma2a(sigmas, assortments):
    nb_col = len(sigmas)
    nb_prod = len(sigmas.T)
    nb_asst = len(assortments)
    ret = np.empty((nb_col, nb_prod, nb_asst), dtype=np.float64)
    for k in range(nb_col):
        ret[k, :, :] = sigma2a(sigmas[k, :], assortments)
    return ret


# defines all the sub-behaviors of rank 1 of the branch sigma in the GDT tree
def add_new_sigma_GDT(sigma_to_duplicate, nb_prod):
    order = len(np.where(sigma_to_duplicate != nb_prod - 1)[0])  # number of ranked products
    # we generate nb_prod-order sigmas in with 'order' at each position of the following mask
    ret = np.empty((0, len(sigma_to_duplicate)), dtype=int)
    for i in list(np.where(sigma_to_duplicate == nb_prod - 1)[0]):
        ret = np.append(ret, [sigma_to_duplicate], axis=0)
    for i in range(nb_prod - order):
        ret[i, np.where(sigma_to_duplicate == nb_prod - 1)[0][i]] = order
    return ret


def compute_eps(A_f, lambda_f, Proba_prod):
    (nb_col, nb_prod, nb_asst) = A_f.shape
    err = 0
    for m in range(nb_asst):
        for i in range(nb_prod):
            err = err + np.abs(np.matmul(A_f[:, i, m], lambda_f) - Proba_prod[m,i])
    return err / (2. * nb_asst)

#############################
