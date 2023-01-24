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

class GP_UCB_SDF(object):
    def __init__(self, f, poisson, beta, noise=0.02, dimension=2, scale=5, 
                 size=50, l=1):
        """
        :param f: the black box function to maximize
        :param poisson: poisson parameter for stochastic delay
        :param beta: a function for i: beta
        :param noise: standard deviation of noise
        :param dimension: dimension of domain
        :param scale: scale of domain
        :param size: size of each axis
        :param l: param for SE_kernel
        """
        # -2.350041373749725 is min value of f1, make minimum value 0 
        self.D = generate_domain(dimension, scale, size)
        self.kernel = SE_kernel(l)
        self.noise = noise
        self.mu = np.array([0. for _ in range(self.D.shape[0])])
        self.sigma = np.array([1. for _ in range(self.D.shape[0])])
        self.poisson = poisson
        self.m = poisson * 2
        self.observe = lambda x: f(x) + np.random.normal(0, self.noise) + 2.350041373749725
        self.beta = beta
        self.f, self.X, self.Y = f, [], []
        self.fmax = f(self.D).max() + 2.350041373749725
        self.fmin = f(self.D).min() + 2.350041373749725
        self.K_inv, self.K_inv_y = None, None
        
        # prune X and Y to remove old censored feedbacks
        self.Xhat = []
        self.Yhat = []
        self.ignore = []

        # samples from poisson
        self.tauT  =  []
        
        logging.info('noise = {}, dimension = {}, scale = {}, size = {}, fmax = {}, l = {}'
                     .format(noise, dimension, scale, size, self.fmax, l))
        logging.info('D shape = {}'.format(self.D.shape))
        logging.info('mu = {}'.format(self.mu))
        logging.info('sigma = {}'.format(self.sigma))

    def update_K(self):
        K = np.array([[self.kernel(x1, x2) for x2 in self.Xhat] for x1 in self.Xhat]) \
            + np.eye(len(self.Xhat)) * (self.noise ** 2)
        L = cholesky(K, lower=True)
        L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        self.K_inv = L_inv.dot(L_inv.T)
        self.K_inv_y = np.dot(self.K_inv, self.Yhat)

    def get_posterior(self, x):
        k_t_x = [self.kernel(_, x) for _ in self.Xhat]
        k_x = self.kernel(x, x)
        sigma = np.sqrt(k_x - np.dot(np.dot(k_t_x, self.K_inv), k_t_x))
        mu = np.dot(k_t_x, self.K_inv_y)
        return mu, sigma

    def update_posterior(self, x_t, censored = False):
        self.X.append(x_t)
        if censored:
            self.Y.append(0)
        else:
            self.Y.append(self.observe(x_t.reshape(1, -1))[0])

        # expurgate self.X and self.Y to give self.Xhat
        # and self.Yhat using self.ignore
        self.Xhat.clear()
        self.Yhat.clear()
        for idx in range(len(self.X)):
            if idx not in self.ignore:
                self.Xhat.append(self.X[idx])
                self.Yhat.append(self.Y[idx])

        self.update_K()
        self.mu, self.sigma = np.array([self.get_posterior(_) for _ in self.D], dtype=float).T

    def delay(self, T):
        ''' returns vector of tauT for every observation'''
        self.tauT = []
        self.delayX = []
        self.delay_idx = []

        for _ in range(T):
            instance = np.random.poisson(self.poisson)
            if instance > self.m:
                self.tauT.append(-1)
            else:
                self.tauT.append(instance)
        
        delay_length = max(max(self.tauT), self.m) + T

        for _ in range(delay_length):
            self.delayX.append([])
            self.delay_idx.append([])

    def run(self, T, output):
        """
        :param T: time horizon
        """
        logging.info('GP_UCB_SDF, T = {}'.format(T))
        
        R_t = 0
        regret = []
        self.delay(T)
        
        for t in range(T):
            x_t = self.D[random_argmax(self.mu + np.sqrt(self.beta(t+1)) * self.sigma)]
            self.update_posterior(x_t, censored = True)
            
            if self.tauT[t] >= 0:
                self.delay_idx[t + self.tauT[t]].append(t)
                self.delayX[t + self.tauT[t]].append(x_t)
            else:
                # if delay is greater than self.m the feedback will always be zero
                pass

            if self.delayX[t]:
                # ignore the zeros of censored feedbacks after delay is over
                self.ignore += self.delay_idx[t]
                for d in range(len(self.delayX[t])):
                    self.update_posterior(self.delayX[t][d])
            
            r_t = self.fmax - self.f(x_t.reshape(1, -1)).reshape(1)[0] - 2.350041373749725
            R_t += r_t
            regret.append([t+1, R_t])
            
            logging.info('t = {}, x_t = {}, r_t = {}, R_t = {};'
                         .format(t+1, x_t, r_t, R_t))
            print(t+1, r_t, R_t)

        pd.DataFrame(regret, columns=['t', 'R_t']).to_csv(output, header=True, index=False)
        x = self.D[random_argmax(self.mu)]
        logging.info('x = {}, f(x) = {}, fmax = {}'.format(x, self.f(x.reshape(1, -1) + 2.350041373749725), self.fmax))