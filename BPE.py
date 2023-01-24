import logging
import random
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

def random_argmax(arr):
    m = arr.max()
    idx = np.nonzero(arr == m)[0]
    return random.choice(idx)

def SE_kernel(l):
    return lambda a, b: np.exp(-np.square(np.linalg.norm(a - b)) / (2 * l ** 2))

def generate_domain(dimension, scale, size):
    meshgrid = np.array(np.meshgrid(*[np.linspace(-scale, scale, size) for _ in range(dimension)]))
    domain = meshgrid.reshape(dimension, -1).T
    return domain

def bped_batch_size(T,Etau,xi,b):
    """returns the batch sizes for BPE-Delay
    
    :param T: time horizon
    :param Etau: Expected value of the tau vector is the poisson parameter
    :param xi: set as 9
    :param b: set as 1
    
    :return: array where the value at index r is the size of the (r+1)th batch
    """
    delta = 0.01
    true_lengths = [1] # q_r values
    batch_size = [] # t_r values 
    while True:
        
        q_r = np.ceil(np.sqrt(T*true_lengths[-1]))
        # t is the point at which we calculate u_delta
        t = sum(batch_size) + q_r
        psi_delta = min( np.sqrt(2*(xi**2)*np.log(1.5*t/delta)) ,  2*b*np.log(1.5*t/delta) )
        u_delta = psi_delta + Etau
        t_r = q_r + u_delta
        
        true_lengths.append(q_r)
        batch_size.append(t_r)
        
        if sum(true_lengths) > T: # last batch
            break

    return np.array(batch_size).astype(int)

def bpe_batch_size(T, delay_T):
    """returns the batch sizes for BPE

    :param T: time horizon
    :param delay_T: the time horizon when adding extensions in BPE-Delay
    
    :return: array of length B where the value at index r is the size of the (r+1)th batch
    """
    
    batch_size = [1] 
    
    while True:
        t_r = np.ceil(np.sqrt(T*batch_size[-1]))
        
        if sum(batch_size) + t_r >= T and sum(batch_size) + t_r < delay_T:
            batch_size.append(t_r)
            batch_size.append(delay_T - sum(batch_size) + 1)
            break
        
        elif sum(batch_size) + t_r >= T and sum(batch_size) + t_r >= delay_T:
            batch_size.append(delay_T - sum(batch_size) + 1)
            break
        
        batch_size.append(t_r)

    return np.array(batch_size[1:]).astype(int)

class BPE(object):
    def __init__(self, f, poisson, bpedelay, beta, noise=0.02, dimension=2, 
                 scale=5, size=50, l=1):
        """
        :param f: the black box function to maximize
        :param poisson: poisson parameter for stochastic delay
        :param bpedelay: boolean if using BPE or BPE-Delay
        :param beta: a function for i: beta
        :param noise: standard deviation of noise
        :param dimension: dimension of domain
        :param scale: scale of domain
        :param size: size of each axis
        :param l: param for SE_kernel
        """
        self.D = generate_domain(dimension, scale, size)
        self.kernel = SE_kernel(l)
        self.noise = noise
        self.poisson = poisson
        self.bpedelay = bpedelay 
        self.mu = np.array([0. for _ in range(self.D.shape[0])])
        self.sigma = np.array([1. for _ in range(self.D.shape[0])])
        self.observe = lambda x: f(x) + np.random.normal(0, self.noise)
        self.beta = beta
        self.f, self.X, self.Y, = f, [], []
        self.fmax = f(self.D).max()
        self.K_inv, self.K_inv_y = None, None
        self.tauT  =  []
        self.stop = False
        
        logging.info('noise = {}, dimension = {}, scale = {}, size = {}, fmax = {}, l = {}'
                     .format(noise, dimension, scale, size, self.fmax, l))
        logging.info('D shape = {}'.format(self.D.shape))
        logging.info('mu = {}'.format(self.mu))
        logging.info('sigma = {}'.format(self.sigma))

    def delay_D(self, last_batch_size, r):
        '''returns observations whose period of delay has ended by the end of the round'''
        if r == 0:
            pass
        else:
            X_hat, Y_hat = [], []
            for i in range(last_batch_size):
                if self.tauT[i] <= last_batch_size:
                    X_hat.append(self.X[i])
                    Y_hat.append(self.Y[i])
            self.X = X_hat
            self.Y = Y_hat

            if self.X:
                print('regular update')
                self.update_K()
                self.mu, self.sigma = np.array([self.get_posterior(_, True) for _ in self.D], dtype=float).T
            elif r==1 and not self.X: # first batch are all delayed
                print('first batch was empty')
                self.sigma = np.array([1. for _ in range(self.D.shape[0])]) # set sigma to the prior
            else:
                print('unexpected result')
                self.stop = True
    
    def update_D(self, r):
        '''pruning the set of potential maximisers'''
        if r == 0:
            pass
        else:
            max_lcb = (self.mu - np.sqrt(self.beta(r)) * self.sigma).max()
            D_idx = np.where(self.mu + np.sqrt(self.beta(r)) * self.sigma >= max_lcb)
            self.D = self.D[D_idx]
            self.mu = self.mu[D_idx]
            self.sigma = self.sigma[D_idx]
        
        logging.info('r = {}, D shape = {}, mu shape = {}, sigma shape = {}'
                     .format(r, self.D.shape, self.mu.shape, self.sigma.shape))

    def update_K(self):
        K = np.array([[self.kernel(x1, x2) for x2 in self.X] for x1 in self.X]) \
            + np.eye(len(self.X)) * (self.noise ** 2)
        L = cholesky(K, lower=True)
        L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        self.K_inv = L_inv.dot(L_inv.T)
        self.K_inv_y = np.dot(self.K_inv, self.Y)

    def get_posterior(self, x, update_mean=False):
        k_t_x = [self.kernel(_, x) for _ in self.X]
        k_x = self.kernel(x, x)
        sigma = np.sqrt(k_x - np.dot(np.dot(k_t_x, self.K_inv), k_t_x))
        if update_mean:
            mu = np.dot(k_t_x, self.K_inv_y)
            return mu, sigma
        else:
            return None, sigma

    def update_posterior(self, x_t, update_mean=False):
        self.X.append(x_t)
        self.Y.append(self.observe(x_t.reshape(1, -1))[0])
        self.update_K()
        if update_mean:
            self.mu, self.sigma = np.array([self.get_posterior(_, True) for _ in self.D], dtype=float).T
        else:
            _, self.sigma = np.array([self.get_posterior(_) for _ in self.D], dtype=float).T

    def tau(self, current_batch_length):
        ''' returns vector of tauT for each
        arm selected in the upcoming round 
        '''
        self.tauT = []
        for i in range(current_batch_length):
            self.tauT.append(i + np.random.poisson(int(self.poisson)))

    def run(self, T, output):
        """
        :param T: time horizon
        """
        Etau = self.poisson
        xi, b = 9, 1
        
        if self.bpedelay:
            batch_size = bped_batch_size(T, Etau, xi, b)
        else:
            batch_size = bpe_batch_size(T, delay_T = 1571)
        
        logging.info(f'T = {T}, batch_sizes = {batch_size}, delay = {bool(self.poisson)}')
        
        t = 1
        R_t = 0
        regret = []
        
        for r in range(len(batch_size)):
            if self.poisson: # we have delay 
                # delay for the last batch
                self.delay_D(batch_size[r-1], r)
                if self.stop: break
                # tau vector for current batch
                self.tau(batch_size[r])
            
            self.update_D(r)
            self.X, self.Y = [], []
            
            for k in range(1, batch_size[r] + 1):
                x_t = self.D[random_argmax(self.sigma)]
                sigma_t = self.sigma.max()
                r_t = self.fmax - self.f(x_t.reshape(1, -1)).reshape(1)[0]
                R_t += r_t

                if k < batch_size[r]:
                    self.update_posterior(x_t)
                else: # last selection of the batch
                    if self.poisson: # we have delay
                        self.X.append(x_t) 
                        self.Y.append(self.observe(x_t.reshape(1, -1))[0])
                    else:
                        self.update_posterior(x_t,update_mean = True)

                print(self.sigma.shape, r_t, R_t)
                regret.append([t, R_t])
                logging.info('r = {}, t_r = {}, k = {}, t = {}, x_t = {}, sigma_t = {}, r_t = {}, R_t = {};'
                             .format(r+1, batch_size[r], k, t, x_t, sigma_t, r_t, R_t))
                t += 1
        
        x = self.D[np.argmax(self.mu)]
        pd.DataFrame(regret, columns=['t', 'R_t']).to_csv(output, header=True, index=False)
        logging.info('x = {}, f(x) = {}, fmax = {}'.format(x, self.f(x.reshape(1, -1)), self.fmax))
