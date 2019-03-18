import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

class random_design:    
    def __init__(self,n,p,Sigma,beta):
        self.Sigma = Sigma
        self.beta = beta
        self.beta_true = beta
        self.n = n
        self.p = p
        self.is_latent = False
        self.is_perturbed = False
        
        L = np.linalg.cholesky(Sigma)
        self.Sigma_sqrt =  L.T
        
        
        self.z = None
        self.y = None 
        self.X = None 
        
    '''
    Some property and metric which only depends on model
    '''
    
    def data_z(self):
        if self.is_latent:
            H = np.random.randn(self.n,self.n_latents)
            X = np.random.randn(self.n,self.p) @ self.Sigma_sqrt + H @self.Gamma
            z = X @ self.beta + H@ self.delta
        elif self.is_perturbed:
            X = np.random.randn(self.n,self.p) @ self.Sigma_sqrt
            z = X @ (self.beta + self.perturbation) 
        else:
            X = np.random.randn(self.n,self.p) @ self.Sigma_sqrt
            z = X @ self.beta 
        self.z = z 
        self.X = X 
        return X,z
    
    def zeros(self):
        return np.zeros(self.p)
    
    def randn(self):
        return np.random.randn(self.p)
    
    
    @property
    def b(self):
        if self.is_latent:
            temp = np.linalg.inv((self.Gamma.T @ self.Gamma + self.Sigma_E ))@ self.Gamma.T @ self.delta
            return temp
        else:
            raise Exception("It is not a latent variable model")
    
    
        
    @property
    def b_l1(self):
        return sum(abs(self.b))
    
    @property
    def beta_l1(self):
        return sum(abs(self.beta))
    
    
    #loss function of the model
    
    def loss(self,a):
        for name_loss in self.name_key_loss:
            print("{}:{} \n".format(name_loss,getattr(self,name_loss)(a)))
            
    def loss_all(self,a):
        for name_loss in self.names_loss:
            print("{}:{} \n".format(name_loss,getattr(self,name_loss)(a)))
            
    
    def angle(self,a):
        temp = np.linalg.norm(a)
        if temp == 0:
            return 0
        else:
            return np.inner(a,self.beta)/(temp*np.linalg.norm(self.beta))
        
    def sin(self,a):
        temp = np.linalg.norm(a)
        if temp == 0:
            return 1
        else:
            cos_theta = np.inner(a,self.beta)/(temp*np.linalg.norm(self.beta))
            return np.sqrt(1- cos_theta**2)
        
    def minus_angle(self,a):
        return - self.angle(a)
    
    def length_ratio(self,a):
        return np.linalg.norm(a)/np.linalg.norm(self.beta)
    
    def jaccard(self,a):
        mask_a = (a!=0)
        mask_beta = (self.beta != 0)
        total = np.sum(np.logical_or(mask_a,mask_beta))
        intersect = np.sum(np.logical_and(mask_a,mask_beta))
        #print(intersect/total)
        return 1 - intersect/total
    
    def l2_par(self,a):
        return sum((a-self.beta)**2)
    
    def l1_par(self,a):
        return sum(abs(a-self.beta))
    
    def l1_par_ratio(self,a):
        return self.l1_par(a)/self.beta_l1
    
    
    def n_true(self,a):
        support = abs(a) > 0.0000000001
        return np.count_nonzero(support * self.beta)
    
    def n_selected(self,a):
        return np.count_nonzero(a)
    
    def FDR(self,a):
        if np.count_nonzero(a) ==0 :
            return 0
        else:
            return 1- self.n_true(a)/np.count_nonzero(a)
    
    def l1_par_ratio(self,a):
        return self.l1_par(a)/self.beta_l1
    
    def l2_pred(self,a):
         return sum((self.X@(a - self.beta_true))**2)/self.n
        
    def accuracy(self,a):
        probability_hat = 1/(1+ np.exp(-self.X@a))
        guessing = probability_hat.round()
        accuracy_vector = 1 - guessing - self.prob + 2*guessing*self.prob
        return sum(accuracy_vector/self.n)
    
    def l2_accuracy(self,a):
        probability_hat = 1/(1+ np.exp(-self.X@a))
        return np.linalg.norm(probability_hat- self.prob)
    
    def getting_ROV(self,BETA,limit = None):
        beta = self.beta
        if type(BETA) is list:
            BETA = np.array(BETA).T
        BETA_mask = (BETA != 0)
        result = []

        beta_mask = (beta != 0 )
        p = beta.shape[0]
        n_true = np.sum(beta_mask)

        for i in range(BETA_mask.shape[1]):
            temp = BETA_mask[:,i]
            TotalD = np.sum(temp)
            TrueD = np.sum(temp*beta_mask)
            if limit is None or TotalD <= limit:
                result.append((TotalD,TrueD))
        result = np.array(result)
        
        return result
    
    def plotting_ROV(self,BETA,limit = None, label = None):
        result = self.getting_ROV(BETA,limit)
        if label is None:
            plt.plot(result[:,0],result[:,1])
        else:
            plt.plot(result[:,0],result[:,1],label = label)
    
    
    
    '''
    Different way to make the model 
    '''
    
    @classmethod 
    def sparse(cls,n,p,s,Sigma = None,key_loss = None):
        beta = np.zeros(p)
        beta[np.random.choice(p,s,replace= False)] = np.random.randn(s)/np.sqrt(s)    
        if Sigma is None:
            Sigma = np.identity(p)
        temp = cls(n,p,Sigma,beta)
        
        if key_loss is not None:
            temp.key_loss = key_loss
        temp.description = temp.description + ",sparse parameter,n:{},p{}".format(n,p)
        return temp
    
    @classmethod
    def sparse_1(cls,n,p,s,size = 1,Sigma = None,key_loss = None):
        beta = np.zeros(p)
        beta[np.random.choice(p,s,replace = False)] = 1/np.sqrt(s)*np.sqrt(size)
        if Sigma is None:
            Sigma = np.identity(p)
        temp = cls(n,p,Sigma,beta)
        if key_loss is not None:
            temp.key_loss = key_loss
            
        temp.description = temp.description + ",sparse parameter,n:{},p{}".format(n,p)
        return temp
    
    @classmethod
    def dense(cls,n,p, Sigma = None):
        beta = np.random.randn(p)*np.sqrt(4)        
        if Sigma is None:
            Sigma = np.identity(p)
        temp = cls(n,p,Sigma,beta)
            
        temp.description = temp.description + ",dense parameter,n:{},p{},".format(n,p) 
        return temp
    
    @classmethod 
    def dense_toep(cls,n,p,rho,key_loss = None):
        
        temp = cls.dense(n,p, Sigma = cls.toep(p,rho))
        temp.description = temp.description + ",toep parameter:{}".format(rho)
        if key_loss is not None:
            temp.key_loss = key_loss
        return temp
    
    @classmethod
    def sparse_toep(cls,n,p,s,rho,key_loss = None):
        temp = cls.sparse(n,p,s,Sigma = cls.toep(p,rho),key_loss = key_loss)
        temp.description = temp.description + ", n:{},p:{},s:{},toep parameter:{}".format(n,p,s,rho)
        return temp
    
    @staticmethod
    def toep(p,rho):
        return(toeplitz([rho**i for i in range(p)]))
    
    @classmethod
    def latent_sparse_toep(cls,n,p,s,rho,n_latents,size_latents,size_delta =1,size_beta = 1,key_loss = None):
        temp = cls.sparse(n,p,s,Sigma = cls.toep(p,rho),key_loss = key_loss)
        temp.beta = temp.beta*np.sqrt(size_beta)

        temp.Gamma = np.random.randn(n_latents,p) /np.sqrt(p) * np.sqrt(size_latents)
        temp.size_latents = size_latents
        
        temp.is_latent = True 
        temp.delta = np.random.randn(n_latents)/np.sqrt(n_latents) * np.sqrt(size_delta)
        temp.n_latents = n_latents
        
        temp.description = "{} latent variables with var {}".format(n_latents,size_latents)
        temp.description += ", n:{},p:{},s{},toep parameter:{}".format(n,p,s,rho)
        
        return temp
    
    @classmethod
    def latent_sparse_toep_1(cls,n,p,s,rho,n_latents,size_latents,size_delta =1,size_beta = 1,key_loss = None):
        temp = cls.sparse_1(n,p,s,Sigma = cls.toep(p,rho),key_loss = key_loss)
        temp.beta = temp.beta*np.sqrt(size_beta)

        temp.Gamma = np.random.randn(n_latents,p) /np.sqrt(p) * np.sqrt(size_latents)
        temp.size_latents = size_latents
        
        temp.is_latent = True 
        temp.delta = np.random.randn(n_latents)/np.sqrt(n_latents) * np.sqrt(size_delta)
        temp.n_latents = n_latents
        
        temp.description = "{} latent variables with var {}".format(n_latents,size_latents)
        temp.description += ", n:{},p:{},s{},toep parameter:{}".format(n,p,s,rho)
        
        return temp
    
    
    
    @classmethod
    def latent_sparse_denormal(cls,n,p,s,rho,n_latents,size_latents,size_delta = 1, size_beta = 1,key_loss = None):
        temp = cls.sparse(n,p,s)
        temp.beta = temp.beta*np.sqrt(size_beta)

        temp.Gamma = np.random.randn(n_latents,p) /np.sqrt(p) * np.sqrt(size_latents)
        temp.size_latents = size_latents
        
        temp.is_perturbed = True 
        temp.Sigma = cls.toep(p,rho) + temp.Gamma.T @ temp.Gamma
        L = np.linalg.cholesky(temp.Sigma)
        temp.Sigma_sqrt =  L.T
        
        temp.delta = np.random.randn(n_latents)/np.sqrt(n_latents) * np.sqrt(size_delta)
        temp.n_latents = n_latents
        
        temp.perturbation =  np.linalg.inv(temp.Gamma.T @temp.Gamma + cls.toep(p,rho))@temp.Gamma.T @ temp.delta
        temp.beta_true = temp.beta+ temp.perturbation
        
        temp.description = "{} latent variables with var {}".format(n_latents,size_latents)
        temp.description += ", n:{},p:{},s{},toep parameter:{}".format(n,p,s,rho)
        return temp
    
    @classmethod
    def latent_noise(cls,n,p,s,rho,n_latents,size_latents,size_delta = 1, size_beta = 1,key_loss = None):
        temp = cls.sparse_1(n,p,s)
        temp.beta = temp.beta
        temp.beta = temp.beta/np.linalg.norm(temp.beta)*np.sqrt(size_beta)

        temp.Gamma = np.random.randn(n_latents,p) /np.sqrt(p) * np.sqrt(size_latents)
        temp.size_latents = size_latents
        
        temp.is_perturbed = True 
        temp.Sigma = cls.toep(p,rho) + temp.Gamma.T @ temp.Gamma
        L = np.linalg.cholesky(temp.Sigma)
        temp.Sigma_sqrt =  L.T
        
        temp.delta = np.random.randn(n_latents)
        temp.n_latents = n_latents
        
        temp.perturbation = np.linalg.inv(temp.Gamma.T @temp.Gamma + cls.toep(p,rho))@temp.Gamma.T @ temp.delta
        temp.perturbation = temp.perturbation/np.linalg.norm(temp.perturbation)*np.sqrt(size_delta)
        temp.beta_true = temp.beta+ temp.perturbation
        
        temp.description = "{} latent variables with var {}".format(n_latents,size_latents)
        temp.description += ", n:{},p:{},s{},toep parameter:{}".format(n,p,s,rho)
        return temp
    
    '''
    Specify the ratio between l2 norm of b and delta
    '''
    @classmethod
    def latent_noise_2(cls,n,p,s,rho,n_latents,size_latents,percentage_noise = 0, size_all = 1,key_loss = None):
        temp = cls.sparse_1(n,p,s)
        temp.beta = temp.beta
        temp.beta = temp.beta/np.linalg.norm(temp.beta)*np.sqrt(size_all*(1-percentage_noise))

        temp.Gamma = np.random.randn(n_latents,p) /np.sqrt(p) * np.sqrt(size_latents)
        temp.size_latents = size_latents
        
        temp.is_perturbed = True 
        temp.Sigma = cls.toep(p,rho) + temp.Gamma.T @ temp.Gamma
        L = np.linalg.cholesky(temp.Sigma)
        temp.Sigma_sqrt =  L.T
        
        temp.delta = np.random.randn(n_latents)
        temp.n_latents = n_latents
        
        temp.perturbation = np.linalg.inv(temp.Gamma.T @temp.Gamma + cls.toep(p,rho))@temp.Gamma.T @ temp.delta
        temp.perturbation = temp.perturbation/np.linalg.norm(temp.perturbation)*np.sqrt(size_all*percentage_noise)
        temp.beta_true = temp.beta+ temp.perturbation
        
        temp.description = "{} latent variables with var {}".format(n_latents,size_latents)
        temp.description += ", n:{},p:{},s{},toep parameter:{}".format(n,p,s,rho)
        return temp
    
    
    '''
    specify the ratio between size of Xb and X delta
    '''
    @classmethod
    def latent_noise_3(cls,n,p,s,rho,n_latents,size_latents,percentage_noise = 0, size_all = 1,key_loss = None):
        temp = cls.sparse_1(n,p,s)
        temp.beta = temp.beta
        temp.beta = temp.beta/np.linalg.norm(temp.beta)*np.sqrt(size_all*(1-percentage_noise))

        temp.Gamma = np.random.randn(n_latents,p) /np.sqrt(p) * np.sqrt(size_latents)
        temp.size_latents = size_latents
        
        temp.is_perturbed = True 
        temp.Sigma = cls.toep(p,rho) + temp.Gamma.T @ temp.Gamma
        L = np.linalg.cholesky(temp.Sigma)
        temp.Sigma_sqrt =  L.T
        
        temp.delta = np.random.randn(n_latents)
        temp.n_latents = n_latents
        

        '''
        Calibrate the size of beta and the perturbation 
        '''
        temp.beta = temp.beta/np.sqrt(temp.beta.T@temp.Sigma@temp.beta)*np.sqrt(size_all*(1-percentage_noise))
        temp.perturbation = np.linalg.inv(temp.Gamma.T @temp.Gamma + cls.toep(p,rho))@temp.Gamma.T @ temp.delta
        temp.perturbation = temp.perturbation/np.sqrt(temp.perturbation.T@temp.Sigma@temp.perturbation)*np.sqrt(size_all*percentage_noise)
        temp.beta_true = temp.beta+ temp.perturbation
        
        temp.description = "{} latent variables with var {}".format(n_latents,size_latents)
        temp.description += ", n:{},p:{},s{},toep parameter:{}".format(n,p,s,rho)
        return temp
    
    
    @classmethod
    def noise_sparse_toep(cls,n,p,s,rho,size_noise,size_signal,key_loss = None):
        """
        For the setting in the paper, where
        Y = X \beta  + X b  + \epsilon
        b is the random noise, 
        and we want to recover the signal \beta
        We generate X with toeplitz Sigma covariance matrix.
        """
        #Constructing design matrix 
        temp = cls.sparse(n,p,s,Sigma = cls.toep(p,rho),key_loss = key_loss)
        temp.beta = temp.beta*np.sqrt(size_signal)
        
               
        temp.is_perturbed = True 
        temp.perturbation = np.random.randn(p) /np.sqrt(p) * np.sqrt(size_noise)
        temp.beta_true = temp.beta+ temp.perturbation
        
        temp.description += "noise with size {}".format(size_noise)
        temp.description += ", n:{},p:{},s{} with random noise ,toep parameter:{},size of noise {}".format(n,p,s,rho,size_noise)
        
        return temp


class linear(random_design):
    def __init__(self,*arg):
        random_design.__init__(self,*arg)
        #self.names_loss = ['l2_par','l1_par','accuracy','n_true','n_selected','FDR','l1_par_ratio','l2_pred']
        self.names_loss = ['l2_par','l1_par','n_true','n_selected','FDR','l1_par_ratio','l2_pred']
        self.name_key_loss = ['l1_par_ratio','l1_par']
        self.key_loss = 'l1_par_ratio'
        self.var =1
        self.description = 'linear model with random design,'
        
    def data(self):
        X,z = self.data_z()
        y = z + np.sqrt(self.var) * np.random.randn(self.n)
        self.y = y
        return X,y
    
    
    
    
    
class logistic(random_design):
    def __init__(self,*arg):
        random_design.__init__(self,*arg)
        self.names_loss = ['l2_par','l1_par','n_true','n_selected','FDR','l1_par_ratio','angle','length_ratio','l2_pred','accuracy','jaccard']
        self.name_key_loss = ['angle','length_ratio']
        self.key_loss = 'minus_angle'
        self.description = 'logistic model with random design,'
        
    def data(self):      
        X,z = self.data_z()
        probability = 1/(1+np.exp(-z))
        self.prob = probability
        y = np.random.binomial(1,probability)
        self.y = y        
        return X,y
    
    
    
class logistic_normal(random_design):
    def __init__(self,*arg):
        random_design.__init__(self,*arg)
        self.var = 1
        self.names_loss = ['l2_pred','l1_par','n_true','n_selected','FDR','l1_par_ratio','angle','length_ratio']
        self.name_key_loss = ['angle','length_ratio']
        self.key_loss = 'minus_angle'
        self.description = 'logistic normal model with random design,'
        
    def data(self):
        X,z = self.data_z()
        y = z + np.random.randn(self.n)* np.sqrt(self.var) 
        probability = 1/(1+np.exp(-y))
        self.prob = probability
        y = np.random.binomial(1,probability)
        self.y = y        
        return X,y
            
    
    def accuracy(self,a):
        return 0
    
    @classmethod 
    def sparse_toep(self,n,p,s,rho,var = 1):
        temp = super(logistic_normal,self).sparse_toep(n,p,s,rho)
        temp.var = var
        return temp
        
    @classmethod
    def dense_toep(self,n,p,rho,var = 1):
        temp = super(logistic_normal,self).dense_toep(n,p,rho)
        temp.var = var
        return temp

class probit(random_design):
    def __init__(self,*arg):
        random_design.__init__(self,*arg)
        self.names_loss = ['l2_par','l1_par','n_true','n_selected','FDR','l1_par_ratio','angle','length_ratio']
        self.name_key_loss = ['angle','length_ratio']
        self.key_loss = 'minus_angle'
        self.description = 'logistic model with random design,'
        
    def data(self):
        X,z = self.data_z()
            
        y = z + np.random.randn(self.n)
        y = y > 0
        y = y.astype(int)
        
        self.y = y
        
        return X,y
    
    def accuracy(self,a):
        return 0


class probit_normal(random_design):
    def __init__(self,*arg):
        random_design.__init__(self,*arg)
        self.names_loss = ['l2_par','l1_par','n_true','n_selected','FDR','l1_par_ratio','angle','length_ratio']
        self.name_key_loss = ['angle','length_ratio']
        self.key_loss = 'minus_angle'
        self.description = 'logistic model with random design,'
        
    def data(self):
        X,z = self.data_z()
        z = z + np.random.randn(self.n)* np.sqrt(self.var)
        
        
        y = z + np.random.randn(self.n)
        y = y > 0
        y = y.astype(int)
        
        self.y = y
        
        return X,y
    
    def accuracy(self,a):
        return 0
    

class poisson(random_design):
    def __init__(self,*arg):
        random_design.__init__(self,*arg)
        self.names_loss = ['l2_par','l1_par','n_true','n_selected','FDR','jaccard', 'l1_par_ratio','angle','length_ratio','l2_pred']
        self.name_key_loss = ['angle','length_ratio']
        self.key_loss = 'minus_angle'
        self.description = 'poisson model with random design,'
        
    def data(self):      
        X,z = self.data_z()
        mean = np.exp(z)
        self.mean = mean
        y = np.random.poisson(mean)
        self.y = y        
        return X,y




