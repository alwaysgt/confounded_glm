from math import inf 
from ._lowlevel import fmin_lbfgs
cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport exp,log,fabs
from numpy cimport ndarray
np.import_array()


'''
recover delta from theta = delta + beta
'''
def recover_delta(beta,alpha_1,alpha_2):
    beta_new = np.zeros(beta.shape)
    threshold = alpha_1/alpha_2 / 2
    for i in range(beta.shape[0]):
        if beta[i] > threshold:
            beta_new[i] = beta[i] - threshold
        elif beta[i] < - threshold:
            beta_new[i] = beta[i] + threshold
        else:
            beta_new[i] = 0
    return beta_new


'''
Get the median of the singular values of argument
'''
def get_sv_median(X):
    D = np.linalg.svd(X,compute_uv = False)
    return D[int(len(D)/2)]


'''
Minus likelihood function of logistic penalized by lava
'''
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double f_lbfgs(np.ndarray[double,ndim = 1] beta,np.ndarray[double,ndim = 1] g,np.ndarray[double,ndim = 2]X, np.ndarray[long,ndim = 1] y,double alpha_1,double alpha_2):
    '''
    Let the first p entries for ridge
    and the last p entries for lasso 
    '''
    cdef int n =  X.shape[0]
    cdef int p =  X.shape[1]
    cdef np.ndarray[double,ndim = 1] xb= np.dot(X,beta)
    
    cdef double temp = 0
    
    '''
    prepare for the loss function 
    '''
    
    cdef double threshold = alpha_1/alpha_2 / 2
    cdef double boundary_constant = alpha_1 * alpha_1 /alpha_2 /4 
    
    
    cdef int i = 0 
    cdef int j = 0
    for j in range(p):
        g[j] = 0
        
    for i in range(n):
        temp -= ( -log(1+ exp(xb[i])) + y[i]*xb[i] )# loglikelihood 
        for j in range(p):
            g[j] -= ((1/(1+ exp(xb[i]))  -1 + y[i])* X[i,j])/n
    
    temp = temp / n
    # Adding penalty
    for i in range(p): 
        if fabs(beta[i]) < threshold:
            g[i] += 2* alpha_2*beta[i]
            temp += alpha_2 * beta[i] * beta[i]
        else:
            temp += boundary_constant + alpha_1 * fabs(fabs(beta[i]) - threshold)
            if beta[i] > 0:
                g[i] += alpha_1
            else:
                g[i] -= alpha_1
    return temp



'''
Minus likelihood function of logistic penalized by ridge
'''
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double f_lbfgs_ridge(np.ndarray[double,ndim = 1] beta,np.ndarray[double,ndim = 1] g,np.ndarray[double,ndim = 2]X, np.ndarray[long,ndim = 1] y,double alpha_2):
    '''
    Let the first p entries for ridge
    and the last p entries for lasso 
    '''
    cdef int n =  X.shape[0]
    cdef int p =  X.shape[1]
    cdef np.ndarray[double,ndim = 1] xb= np.dot(X,beta)
    
    cdef double temp = 0
    
    '''
    prepare for the loss function 
    '''
    
    
    cdef int i = 0 
    cdef int j = 0
    for j in range(p):
        g[j] = 0
        
    for i in range(n):
        temp -= ( -log(1+ exp(xb[i])) + y[i]*xb[i] )# loglikelihood 
        for j in range(p):
            g[j] -= ((1/(1+ exp(xb[i]))  -1 + y[i])* X[i,j])/n
    
    temp = temp / n
    # Adding penalty
    for i in range(p): 
        g[i] += 2* alpha_2*beta[i]
    return temp


    
'''
Solve the penalized MLE of logistic
''' 
def optimize(X,y,alpha_1,alpha_2,beta =None):
    n,p = X.shape
    if beta is None:
        beta = np.zeros(X.shape[1])
    if alpha_1 is 0:
        return fmin_lbfgs(f_lbfgs_ridge, beta, args = (X,y,alpha_2),
                           max_linesearch = 100)#,min_step = 1e-40)
    else:
        return fmin_lbfgs(f_lbfgs, beta, args = (X,y,alpha_1,alpha_2),
                           max_linesearch = 100,min_step = 1e-40)
                           


    
    
'''
Percentage LAVA with a exact quantile of the singular values
'''

class lava_logistic_percentage_adaptive:
    def __init__(self,percentage):
        self.percentage = percentage
        
    def regress(self,X,y,model, loss = None):    
        N = 100
        percentage = self.percentage
        _,D,_ = np.linalg.svd(X)
        n,p = X.shape
        alpha_2 =  D[int(len(D)*percentage)]**2/n
        

        factor = 1

        beta4 = optimize(X,y,0,factor*alpha_2)
        lambda_max = np.max(np.abs(beta4)) * 2 * alpha_2

        if loss is None:
            loss = model.key_loss
        oracle = getattr(model,loss)

        l1 = inf
        beta_best = None
        best_indx  = None
        i = 0

        beta = np.zeros(p)
        for n in range(N):
            factor = self.get_factor(X,beta)**2
            alpha_1 = np.exp(np.log(0.0001)/N*n) * lambda_max
            temp = optimize(X,y, alpha_1,factor *alpha_2,beta= beta)
            temp_delta = recover_delta(temp,alpha_1,factor * alpha_2)
            
            #print(factor)

            #check whether the objective improves
            if loss =="l2_pred":
                temp2 = oracle(temp_delta)
            elif loss == "accuracy":
                temp2 = -oracle(temp_delta)
            else:
                temp2 = oracle(temp_delta)
            if temp2 < l1:
                beta_best = temp_delta
                #print(i,":",oracle(temp[p:]))
                l1 = temp2        
                best_indx = alpha_1
            beta = temp 
        return beta_best
    
    def regress_result(self,X,y):    
        N = 100
        percentage = self.percentage
        _,D,_ = np.linalg.svd(X)
        n,p = X.shape
        alpha_2 =  D[int(len(D)*percentage)]**2/n
        

        factor = 1

        beta4 = optimize(X,y,0,factor*alpha_2)
        lambda_max = np.max(np.abs(beta4)) * 2 * alpha_2


        beta = np.zeros(p)
        result = []
        for n in range(N):
            factor = self.get_factor(X,beta)**2
            alpha_1 = np.exp(np.log(0.0001)/N*n) * lambda_max
            temp = optimize(X,y,alpha_1,factor * alpha_2,beta= beta)
            temp_delta = recover_delta(temp,alpha_1,factor * alpha_2)
            result.append(temp_delta)
            beta = temp
        return result
    
    def get_factor(self,X,beta):
        xb = X@beta
        temp  = np.exp(xb)
        return np.mean(np.sqrt(temp)/(1+temp))

    



