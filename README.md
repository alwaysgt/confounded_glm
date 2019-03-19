Confounded_glm package is written for the thesis [Spectral Deconfounding on Generalized Linear Models](https://owgt.me/deconfounding_lava.html). It implements the deconfounding method introduced in the thesis and provides module to simulate data from confounded glms model. 

The deconfounding method is provided in the module `confounded_glm.fitting` and the simulation function is provided in the module `confounded_glm.model`.



# Usage
## confounded_glm.model
We show how to generate simulate data with this module.

```python
import confounded_glm.model
# number of the samples
n = 300 
# number of the observed variables
p = 300
# sparsity of beta_0
s = 10
# argument that specify the correlation of observed variables X. 
# Cov(X) is set to be a toeplitz matrix with parameter rho.
rho = 0.1 
# number of latent variables
n_latent = 10 
# argument that specify the size of the effect of the latent variables on the dependent variable.
size_delta = 0.1

model  = confounded_glm.model.logistic.latent_sparse_toep(n,p,s,rho,n_latent,size_delta)
X,y = model.data()
```


## confounded_glm.fitting 
Given data X,y, we show how to implement our methods on the data. 
```python
import confounded_glm.fitting

## Getting the estimation for a single lambda_1
percentage = 0.5 
alpha_1 = 0.05
delta,beta = confounded_glm.fitting.deconfounding_lava_logistic(X,y,percentage, lambda_1)

## Getting the estimation for a sequence of lambda_1
path = confounded_glm.fitting.deconfounding_lava_logistic_path(X,y,percentage)
```
