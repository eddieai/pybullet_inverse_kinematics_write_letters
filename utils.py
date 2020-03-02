from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import numpy as np

from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as normal_dist


from numpy import dot
from numpy.linalg import inv
from numpy.linalg import pinv

import time
from scipy.stats import multivariate_normal as mvn

class GMM():
    def __init__(self, D = 1, K = 2,  reg_factor = 1e-6):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        self.reg_factor =  reg_factor 

        
    def init_kmeans(self):
        kMM = KMeans(n_clusters=self.K).fit(self.x)
        self.means_ = kMM.cluster_centers_
        for i in range(self.K):
            self.covariances_[i] = np.cov(self.x[kMM.labels_==i].T) + np.eye(self.D)*self.reg_factor
        
    def init_random(self):
        self.means_ = self.x[np.random.choice(len(self.x),size = self.K)]
        for i in range(self.K):
            self.covariances_[i] = np.cov(self.x.T)

    def fit(self,x, max_iter = 10, init_type = 'kmeans', threshold = 1e-4, n_init = 5):
        self.x = x
        self.N = len(self.x) #number of datapoints
        self.Ls = np.zeros((self.N,self.K)) #posterior probability of z
        self.zs = np.zeros((self.N,self.K)) #posterior probability of z

        self.threshold = threshold
        
        best_params = ()
        Lmax = -np.inf
        for it in range(n_init):
            if init_type == 'kmeans':
                self.init_kmeans()
            elif init_type == 'random':
                self.init_random()

            for i in range(max_iter):
                print('Iteration ' + str(i))
                self.expectation()
                self.maximization()
                print(self.L)
                if np.abs(self.prev_L-self.L) < self.threshold:
                    break
                    
            if self.L > Lmax:
                Lmax = self.L
                best_params = (self.L, self.weights_.copy(), self.means_.copy(), self.covariances_.copy(), self.zs.copy(), self.Ns.copy())
            
        #return the best result
        self.L = Lmax
        self.weights_ = best_params[1]
        self.means_ = best_params[2]
        self.covariances_ = best_params[3]
        self.zs = best_params[4]
        self.Ns = best_params[5]
        print('Obtain best result with Log Likelihood: ' + str(self.L))
        
    def expectation(self):
        for k in range(self.K):
            self.Ls[:,k] = self.weights_[k]*mvn.pdf(self.x,mean = self.means_[k], cov=self.covariances_[k])

        self.zs = self.Ls/np.sum(self.Ls,axis=1)[:,None] #normalize

        self.prev_L = self.L
        self.L = np.sum(np.log(np.sum(self.Ls, axis=1)))/self.N
        self.Ns = np.sum(self.zs,axis=0)
             
    def maximization(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 

            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]
            
            #update covariance
            x_reduce_mean = self.x-self.means_[k,:]
            #S_k = dot(x_reduce_mean.T, dot(np.diag(self.zs[:,k]), x_reduce_mean))
            sigma_k = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            
            self.covariances_[k,:] = sigma_k        
            
    def get_marginal(self,dim,dim_out=None):
        if dim_out is not None:
            means_, covariances_ = (self.means_[:,dim],self.covariances_[:,dim,dim_out])
        else:
            means_, covariances_ = (self.means_[:,dim],self.covariances_[:,dim,dim])
        return means_,covariances_
    
    def condition(self,x_in,dim_in,dim_out,h=None, return_gmm = False):
        mu_in, sigma_in = self.get_marginal(dim_in)
        mu_out, sigma_out = self.get_marginal(dim_out)
        _, sigma_in_out = self.get_marginal(dim=dim_in, dim_out = dim_out)
        
        if h is None:
            h = np.zeros(self.K)
            for k in range(self.K):
                h[k] = self.weights_[k]*mvn(mean=mu_in[k],cov=sigma_in[k]).pdf(x_in)
            h = h/np.sum(h) 
        
        #compute mu and sigma
        mu = []
        sigma = []
        for k in range(self.K):
            mu += [mu_out[k] + np.dot(sigma_in_out[k].T, np.dot(np.linalg.inv(sigma_in[k]), (x_in-mu_in[k]).T)).flatten()]
            sigma += [sigma_out[k] - np.dot(sigma_in_out[k].T, np.dot(np.linalg.inv(sigma_in[k]), sigma_in_out[k]))]
            
        mu,sigma = (np.asarray(mu),np.asarray(sigma))
        if return_gmm:
            return h,mu,sigma
        else:
            return self.moment_matching(h,mu,sigma)
        
    def moment_matching(self,h,mu,sigma):
        dim = mu.shape[1]
        sigma_out = np.zeros((dim, dim))
        mu_out = np.zeros(dim)
        for k in range(self.K):
            sigma_out += h[k]*(sigma[k] + np.outer(mu[k],mu[k]))
            mu_out += h[k]*mu[k]
            
        sigma_out -= np.outer(mu_out, mu_out)
        return mu_out,sigma_out
        
    def plot(self):
        fig,ax = plt.subplots()
        plot_GMM(self.means_, self.covariances_, ax)
        

class HDGMM(GMM):
    def __init__(self, D = 1, K = 2,  reg_factor = 1e-6, n_fac = 1):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        self.reg_factor =  reg_factor 
        self.n_fac = n_fac
             
    def maximization(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 

            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]
            
            #update covariance
            x_reduce_mean = self.x-self.means_[k,:]
            sigma_k = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            
            #modify the covariance by replacing the last (D-n_fac) eigen values by their average
            D,V = np.linalg.eig(sigma_k)
            sort_indexes = np.argsort(D)[::-1]
            D = np.concatenate([D[sort_indexes[:self.n_fac]], [np.mean(D[sort_indexes[self.n_fac:]])]*(self.D-self.n_fac)])
            V = V[:,sort_indexes]
            self.covariances_[k,:] = dot(V, dot(np.diag(D), V.T))+ np.eye(self.D)*self.reg_factor       
            
class semitiedGMM(GMM):
    def __init__(self, D = 1, K = 2,  reg_factor = 1e-6, bsf_param =  5E-2, n_step_variation = 50):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        self.reg_factor =  reg_factor 
        self.bsf_param = bsf_param
        self.n_step_variation = n_step_variation
        self.B = np.eye(self.D)*self.bsf_param
        self.S = np.array([np.eye(D) for i in range(K)])
        self.Sigma_diag = np.array([np.eye(D) for i in range(K)])
        #def init_semitiedGMM(self):
        #self.H_init = pinv(self.B) + np.eye(self.D)*self.reg_factor
        #self.Sigma_diag_init = np.array([np.eye(self.D) for i in range(K)])
        #for i in range(self.K):
        #    eig_vals, V = np.linalg.eig(self.covariances_[i])
        #    self.Sigma_diag_init[i] = np.diag(eig_vals)
            
    def maximization(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 

            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]
            
            #calculate the sample covariances
            x_reduce_mean = self.x-self.means_[k,:]
            self.S[k] = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            
        #calculate H and the covariance
        for it in range(self.n_step_variation):
            for k in range(self.K):
                self.Sigma_diag[k] = np.diag(np.diag(dot(self.B, dot(self.S[k], self.B.T))))
                
            for i in range(self.D):
                C = pinv(self.B.T)*np.linalg.det(self.B)
                G = np.zeros((self.D,self.D))
                for k in range(self.K):
                    G += dot(self.S[k], np.sum(self.zs[:,k]))/self.Sigma_diag[k, i, i]
                self.B[i] = dot(C[i],pinv(G))*np.sqrt(np.sum(self.zs)/dot(C[i], dot(pinv(G), C[i].T)))
        self.H = pinv(self.B) + np.eye(self.D)*self.reg_factor
        for k in range(self.K):
            self.covariances_[k,:] = dot(self.H, dot(self.Sigma_diag[k], self.H.T))   
        
        
class MFA(GMM):
    def __init__(self, D = 1, K = 2, n_fac = 1, reg_factor = 1e-6):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        

        
        self.Lambda_ = np.array([np.zeros((D,n_fac)) for i in range(K)])
        self.Psi_ = np.array([np.eye(D) for i in range(K)])
        self.n_fac = n_fac
        self.reg_factor = reg_factor
            
    def init_MFA(self):
        for k in range(self.K):
            self.Psi_[k] = np.diag(np.diag(self.covariances_[k]))
            D,V = np.linalg.eig(self.covariances_[k] - self.Psi_[k])
            indexes = np.argsort(D)[::-1]
            V = dot(V[:,indexes], np.diag(np.lib.scimath.sqrt(D[indexes])))
            self.Lambda_[k] = V[:,:self.n_fac]
            
            B_k = dot(self.Lambda_[k].T, inv(dot(self.Lambda_[k],self.Lambda_[k].T)+self.Psi_[k]))
            self.Lambda_[k] = dot(self.covariances_[k],dot(B_k.T, inv(np.eye(self.n_fac)- dot(B_k,self.Lambda_[k]) + dot(B_k,dot(self.covariances_[k],B_k.T)))))
            self.Psi_[k] = np.diag(np.diag(self.covariances_[k] - dot(self.Lambda_[k], dot(B_k,self.covariances_[k])))) + np.eye(self.D)*self.reg_factor
           

    def fit(self,x, max_iter = 10, init_type = 'kmeans', threshold = 1e-4, n_init = 5):
        self.x = x
        self.N = len(self.x) #number of datapoints
        self.threshold = threshold
        
        self.Ls = np.zeros((self.N,self.K)) #posterior probability of z
        self.zs = np.zeros((self.N,self.K)) #posterior probability of z

        
        best_params = ()
        Lmax = -np.inf

        for it in range(n_init):
            if init_type == 'kmeans':
                self.init_kmeans()
            elif init_type == 'random':
                self.init_random()

            self.init_MFA()
                
            for i in range(max_iter):
                print('Iteration ' + str(i))
                
                tic = time.time()
                self.expectation()
                toc = time.time()
                #print('Expectation computation', toc-tic)
            
                tic = time.time()
                self.maximization_1()
                toc = time.time()
                #print('Maximization 1 computation', toc-tic)

                #self.expectation()
                
                tic = time.time()
                self.maximization_2()
                toc = time.time()
                #print('Maximization 2 computation', toc-tic)

                print(self.L)
                if np.abs(self.prev_L-self.L) < self.threshold:
                    break
                    
            if self.L > Lmax:
                Lmax = self.L
                best_params = (self.L, self.weights_.copy(), self.means_.copy(), self.covariances_.copy(), self.zs.copy(), self.Ns.copy())
            
        #return the best result
        self.L = Lmax
        self.weights_ = best_params[1]
        self.means_ = best_params[2]
        self.covariances_ = best_params[3]
        self.zs = best_params[4]
        self.Ns = best_params[5]
        print('Obtain best result with Log Likelihood: ' + str(self.L))

    def maximization_1(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 
            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]  

    def maximization_2(self):
        #update covariance
        for k in range(self.K):
            x_reduce_mean = self.x-self.means_[k,:]
            #S_k = dot(x_reduce_mean.T, dot(np.diag(self.zs[:,k]), x_reduce_mean))
            S_k = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            B_k = dot(self.Lambda_[k].T, inv(dot(self.Lambda_[k],self.Lambda_[k].T)+self.Psi_[k]))
            
            
            self.Lambda_[k] = dot(S_k,dot(B_k.T, inv(np.eye(self.n_fac)- dot(B_k,self.Lambda_[k]) + dot(B_k,dot(S_k,B_k.T)))))
            self.Psi_[k] = np.diag(np.diag(S_k - dot(self.Lambda_[k], dot(B_k,S_k)))) + np.eye(self.D)*self.reg_factor
            self.covariances_[k] = dot(self.Lambda_[k], self.Lambda_[k].T) + self.Psi_[k]
        

class MPPCA(GMM):
    def __init__(self, D = 1, K = 2, n_fac = 1, reg_factor = 1e-6):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        
        self.Lambda_ = np.array([np.zeros((D,n_fac)) for i in range(K)])
        self.Psi_ = np.array([np.eye(D) for i in range(K)])
        self.sigma_ = np.ones(self.K)*1e-4
        
        self.n_fac = n_fac
        self.reg_factor = reg_factor
        

            
    def init_MPPCA(self):
        for k in range(self.K):
            self.sigma_[k] = np.trace(self.covariances_[k])/self.D
            print(self.sigma_[k])
            D,V = np.linalg.eig(self.covariances_[k] - np.eye(self.D)*self.sigma_[k])
            indexes = np.argsort(D)[::-1]
            print(D)
            V = dot(V[:,indexes], np.diag(np.lib.scimath.sqrt(D[indexes])))
            self.Lambda_[k] = V[:,:self.n_fac]
                       

    def fit(self,x, max_iter = 10, init_type = 'kmeans', threshold = 1e-4, n_init = 5):
        self.x = x
        self.N = len(self.x) #number of datapoints
        self.Ls = np.zeros((self.N,self.K)) #posterior probability of z
        self.zs = np.zeros((self.N,self.K)) #posterior probability of z

        self.threshold = threshold
        
        best_params = ()
        Lmax = -np.inf

        for it in range(n_init):
            if init_type == 'kmeans':
                self.init_kmeans()
                print(self.means_)
                print(self.covariances_)
                print(self.sigma_)
            elif init_type == 'random':
                self.init_random()

            self.init_MPPCA()
            print(self.sigma_)
                
            for i in range(max_iter):
                print('Iteration ' + str(i))
                
                #tic = time.time()
                self.expectation()
                #toc = time.time()
                #print('Expectation computation', toc-tic)
            
                #tic = time.time()
                self.maximization_1()
                #toc = time.time()
                #print('Maximization 1 computation', toc-tic)

                #self.expectation()
                
                #tic = time.time()
                self.maximization_2()
                #toc = time.time()
                #print('Maximization 2 computation', toc-tic)

                print(self.L)
                if np.abs(self.prev_L-self.L) < self.threshold:
                    break
                    
            if self.L > Lmax:
                Lmax = self.L
                best_params = (self.L, self.weights_.copy(), self.means_.copy(), self.covariances_.copy(), self.zs.copy(), self.Ns.copy())
            
        #return the best result
        self.L = Lmax
        self.weights_ = best_params[1]
        self.means_ = best_params[2]
        self.covariances_ = best_params[3]
        self.zs = best_params[4]
        self.Ns = best_params[5]
        print('Obtain best result with Log Likelihood: ' + str(self.L))

    def maximization_1(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 
            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]  

    def maximization_2(self):
        #update covariance
        self.S = []
        self.M = []

        for k in range(self.K):
            x_reduce_mean = self.x-self.means_[k,:]
            #S_k = dot(x_reduce_mean.T, dot(np.diag(self.zs[:,k]), x_reduce_mean))
            S_k = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            M_k = dot(self.Lambda_[k].T,self.Lambda_[k]) + np.eye(self.n_fac)*self.sigma_[k]
            
            Lambda_k = dot(S_k,dot(self.Lambda_[k], inv(np.eye(self.n_fac)*self.sigma_[k] + \
                                dot(inv(M_k),dot(self.Lambda_[k].T,dot(S_k, self.Lambda_[k]) )  ))))
            self.S.append(S_k)
            self.M.append(M_k)
            
            
            self.sigma_[k] = np.trace(S_k-dot(S_k,dot(self.Lambda_[k],dot(inv(M_k),Lambda_k.T))))/self.D
            self.Psi_[k] = np.eye(self.D)*self.sigma_[k]
            
            self.Lambda_[k] = Lambda_k.copy()
            
            self.covariances_[k] = dot(self.Lambda_[k], self.Lambda_[k].T) + self.Psi_[k]
                   
        
class GMR():
    def __init__(self, GMM, n_in, n_out):
        self.GMM = GMM
        self.n_in = n_in
        self.n_out = n_out
        #segment the gaussian components
        self.mu_x = []
        self.mu_y = []
        self.sigma_xx = []
        self.sigma_yy = []
        self.sigma_xy = []
        self.sigma_xyx = []
        for k in range(self.GMM.n_components):
            self.mu_x.append(self.GMM.means_[k][0:self.n_in])        
            self.mu_y.append(self.GMM.means_[k][self.n_in:])        
            self.sigma_xx.append(self.GMM.covariances_[k][0:self.n_in, 0:self.n_in])        
            self.sigma_yy.append(self.GMM.covariances_[k][self.n_in:, self.n_in:])        
            self.sigma_xy.append(self.GMM.covariances_[k][0:self.n_in, self.n_in:])
            self.sigma_xyx.append(np.dot(self.sigma_xy[k].T,np.linalg.inv(self.sigma_xx[k])))
            
        self.mu_x = np.array(self.mu_x)
        self.mu_y = np.array(self.mu_y)
        self.sigma_xx = np.array(self.sigma_xx)
        self.sigma_yy = np.array(self.sigma_yy)
        self.sigma_xy = np.array(self.sigma_xy)
        self.sigma =[self.sigma_yy[k]- np.dot(self.sigma_xy[k].T, \
            np.dot(np.linalg.inv(self.sigma_xx[k]), self.sigma_xy[k])) for k in range(self.GMM.n_components)]
        
    def predict(self,x):
        h = []
        mu = []        

        for k in range(self.GMM.n_components):
            h.append(self.GMM.weights_[k]*mvn(mean = self.mu_x[k], cov = self.sigma_xx[k]).pdf(x))
            mu.append(self.mu_y[k] + np.dot(self.sigma_xyx[k], x - self.mu_x[k]))
        
        h = np.array(h)
        h = h/np.sum(h)
        mu = np.array(mu)
        sigma = self.sigma
        
        sigma_one = np.zeros([self.n_out, self.n_out])
        mu_one = np.zeros(self.n_out)
        for k in range(self.GMM.n_components):
            sigma_one += h[k]*(sigma[k] + np.outer(mu[k],mu[k]))
            mu_one += h[k]*mu[k]
            
        sigma_one -= np.outer(mu_one, mu_one)
        return mu_one, sigma_one

def plot_gaussian_1D(mu, sigma, ax,offset = None, bound= None, color = [.4,.4,.4], alpha = 1., normalize = True, prior = None, label = 'label', orient = 'h'):
    n = 100
    if bound is None:
        bound = [mu-2*sigma, mu+2*sigma]
        
    x = np.linspace(bound[0], bound[1], n)
    y = normal_dist(loc=mu, scale=sigma).pdf(x)
    if normalize:
        y = y/np.max(y)
    if prior is not None:
        y *= prior

    if offset is not None:
        y += offset
    
    if orient == 'h':
        poly_data = np.vstack([x,y]).T
        axis_limit = [bound[0], bound[1], 0, np.max(y)]
    else:
        poly_data = np.vstack([y,x]).T
        axis_limit = [0, np.max(y), bound[0], bound[1]]
        
    polygon = Polygon(poly_data,False,color=color,label=label, alpha=alpha)
    #plt.plot(y,x)
    ax.add_patch(polygon)

    plt.axis(axis_limit)
    return x,y

def plot_dist_1D(x,y,ax, color = [.4,.4,.4], alpha = 1.,label = 'label'):
    poly_data = np.vstack([x,y]).T
    axis_limit = [bound[0], bound[1], 0, np.max(y)]
    polygon = Polygon(poly_data,False,color=color,label=label, alpha=alpha)
    ax.add_patch(polygon)
    plt.axis(axis_limit)
    return

def plot_gaussian_2D(mu, sigma,ax,color=[0.7,0.7,0.7],alpha=1.0, label='label'):
    eig_val, eig_vec = np.linalg.eigh(sigma)
    std = np.sqrt(eig_val)*2
    angle = np.arctan2(eig_vec[1,0],eig_vec[0,0])
    ell = Ellipse(xy = (mu[0], mu[1]), width=std[0], height = std[1], angle = np.rad2deg(angle))
    ell.set_facecolor(color)
    ell.set_alpha(alpha)
    ell.set_label(label)
    ax.add_patch(ell)
    return


def plot_data_2D(x, y_true, y_pred, colors = ['r', 'k', 'b'], title = 'Linear regression', alphas = None):
    if alphas is None:
        alphas = np.ones(len(x))
    #plot the predicted data
    plt.plot(x,y_pred, '-' + colors[0])
    
    #plot the error bar and the true data
    for i in range(len(x)):
        #plot the true data
        plt.plot(x[i],y_true[i], '.' + colors[1])#,alpha = alphas[i])
        plt.plot([x[i],x[i]], [y_true[i], y_pred[i]], '-'+colors[2], alpha = alphas[i])
    plt.xlabel(u'x\u2081')
    plt.ylabel(u'y\u2081')
    
    plt.title(title)
    
    return

def plot_GMM(mus, sigmas, ax, colors = None, alphas = None, labels = None):
    n = len(mus)
    if colors is None:
        colors = [[0.7,0.7,0.7]]*n
        
    for i in range(n):
        if labels is None:
            plot_gaussian_2D(mus[i], sigmas[i], ax, color=colors[i])
        else:
            plot_gaussian_2D(mus[i], sigmas[i], ax,label=labels[i],color=colors[i])
    return

import matplotlib.pyplot as plt 

def plot_with_covs_1D(x, y, cov, ax):
    y_low = y - 2*np.sqrt(cov)
    y_up = y + 2*np.sqrt(cov)
    y_up = y_up[::-1]
    
    x_1 = np.concatenate([x, x[::-1]])
    y_1 = np.concatenate([y_low, y_up])
    xy = np.vstack([x_1,y_1]).T
    poly = Polygon(xy,alpha=0.4)
    ax.add_patch(poly)
    
    plt.plot(x,y,'-r')