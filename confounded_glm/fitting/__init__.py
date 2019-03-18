from ._lowlevel import fmin_lbfgs
from ._fit import optimize
import numpy as np



'''
Recover delta from theta given lambda_1 and alpha2
'''
def recover_delta(theta,lambda_1,lambda_2):
    delta = np.zeros(theta.shape)
    threshold = lambda_1/lambda_2 / 2
    for i in range(theta.shape[0]):
        if theta[i] > threshold:
            delta[i] = theta[i] - threshold
        elif theta[i] < - threshold:
            delta[i] = theta[i] + threshold
        else:
            delta[i] = 0
    return delta


'''
Calculating a proper lambda_2 for deconfounding.
We provide two function. One calculates approximately and one does exactly.
'''
def get_lambda_2_approx(X,theta,percentage,lambda_2_old):
    xb = X@theta
    temp  = np.exp(xb)
    return lambda_2_old*(np.mean(np.sqrt(temp)/(1+temp)))**2 

def get_lambda_2_exact(X,theta,percentage,lambda_2_old=None):
    xb = X@theta
    temp  = np.exp(xb)
    inverse_variance_matrix = np.diag(np.sqrt(temp)/(1+temp))
    D = np.linalg.svd(inverse_variance_matrix@X,compute_uv =False)
    n,p = X.shape
    lambda_2 =  D[int(len(D)*percentage)]**2/n
    return lambda_2







'''
Solving the deconfounding lava estimator with a given percentage and lambda_1.
We use the idea to accelerate, which is discussed in Sec 4.3.2 in the thesis
'''
def deconfounding_lava_logistic(X,y,percentage,lambda_1,approx=True):
    _,D,_ = np.linalg.svd(X)
    n,p = X.shape
    theta = np.zeros(p)
    lambda_2_old =  get_lambda_2_exact(X,theta,percentage)
    if approx:
        get_lambda_2 = get_lambda_2_approx
    else:
        get_lambda_2 = get_lambda_2_exact

    lambda_2 = get_lambda_2(X,theta,percentage,lambda_2_old)
    theta = optimize(X,y,lambda_1,lambda_2)
    lambda_2_old
    while(1):
        lambda_2 = get_lambda_2(X,theta,percentage,lambda_2_old)
        theta_new = optimize(X,y,lambda_1,lambda_2,theta)
        if np.sum(np.abs(theta- theta_new))/np.sum(np.abs(theta)) < 1e-3:
            theta = theta_new
            break
        theta = theta_new
    delta = recover_delta(theta, lambda_1, lambda_2)
    return delta,theta - delta





'''
Solving the regularization path with a given percentage
'''
def deconfounding_lava_logistic_path(X,y,percentage,approx = True):
    N = 100
    _,D,_ = np.linalg.svd(X)
    n,p = X.shape
    lambda_2_old =  D[int(len(D)*percentage)]**2/n
    if approx:
        get_lambda_2 = get_lambda_2_approx
    else:
        get_lambda_2 = get_lambda_2_exact
        

    theta4 = optimize(X,y,0,4*lambda_2_old)
    lambda_max = np.max(np.abs(theta4)) * 2 * lambda_2_old


    i = 0

    theta = np.zeros(p)
    result= []
    for n in range(N):
        lambda_2 = get_lambda_2(X,theta,percentage,lambda_2_old)
        lambda_1 = np.exp(np.log(0.0001)/N*n) * lambda_max
        temp = optimize(X,y, lambda_1,lambda_2,theta)
        temp_delta = recover_delta(temp, lambda_1, lambda_2)
        result.append((temp_delta,temp - temp_delta))
        theta = temp
    return result
    

    
    




