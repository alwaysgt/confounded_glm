import numpy as np
'''
Generate data
'''

import confounded_glm.model

# number of the samples
n = 300 
# number of the observed variables
p = 100
# sparsity of beta_0
s = 5
# argument that specify the correlation of observed variables X. 
# Cov(X) is set to be a toeplitz matrix with parameter rho.
rho = 0.3
# number of latent variables
n_latent = 10 
# argument that specify the size of the effect of the latent variables on the dependent variable.
size_delta = 1

model  = confounded_glm.model.logistic.latent_sparse_toep(n,p,s,rho,n_latent,size_delta)
X,y = model.data()


'''
fitting the model
'''
import confounded_glm.fitting

## Getting the estimation for a single lambda_1
percentage = 0.5 
lambda_1 = 0.01
delta,beta = confounded_glm.fitting.deconfounding_lava_logistic(X,y,percentage, lambda_1)

## Getting the estimation for a sequence of lambda_1
path = confounded_glm.fitting.deconfounding_lava_logistic_path(X,y,percentage)
