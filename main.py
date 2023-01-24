import datetime as dt
import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import BPE
import GP_UCB
import GP_UCB_SDF

now = dt.datetime.now().strftime("%Y%m%d%H%M%S")
BETA = {'6': lambda i: 6} # can add a variety of betas

class function(object):
    def __init__(self, name, l):
        self.points = pd.read_csv('{}.csv'.format(name))
        self.gpr = GaussianProcessRegressor(kernel=RBF(l), random_state=0)
        self.gpr.fit(self.points[['x_1', 'x_2']].values, self.points[['y']].values)

    def __call__(self, X):
        return self.gpr.predict(X)

if __name__ == '__main__':
    T = 1000
    f = sys.argv[1]  # name given to the objective function e.g. 'f1'
    beta_str = sys.argv[2]  # beta value, we use 6 in all experiments
    delay = sys.argv[3] # poisson parameter for the stochastic delay
    algo = sys.argv[4] # BPE or BPE-Delay or UCB or UCB-SDF
    
    beta = BETA[beta_str]
    DIR = os.getcwd()
    folder = '{}_{}_{}_{}'.format(f, beta_str, delay, algo)
    os.makedirs(folder)
    func = function(f, 1) # length scale parameter to generate the objective function

    # 10 independent runs
    for i in range(10):
        LOG_PATH = os.path.join(DIR, folder,
                                '{}_{}_{}_{}_{}_{}_{}.log'.format(T, f, beta_str, algo, delay, now, i))
        REGRET_PATH = os.path.join(DIR, folder,
                                    '{}_{}_{}_{}_{}_{}_{}.csv'.format(T, f, beta_str, algo, delay, now, i))
        
        logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
        logging.info('beta = {}'.format(sys.argv[2:]))

        if algo in ['BPE', 'bpe']:
            bo = BPE.BPE(func, poisson = int(delay), bpedelay = False, beta=beta, l=1)
            bo.run(T, REGRET_PATH)
        elif algo in ['BPE-Delay', 'bpe-delay', 'bpedelay']:
            bo = BPE.BPE(func, poisson = int(delay), bpedelay = True, beta=beta, l=1)
            bo.run(T, REGRET_PATH)
        elif algo in ['ucb', 'UCB']:
            bo = GP_UCB.GP_UCB(func, poisson = int(delay),  beta=beta, l=1)
            bo.run(T, REGRET_PATH)
        elif algo in ['UCB-SDF', 'ucb-sdf']:
            bo = GP_UCB_SDF.GP_UCB_SDF(func, poisson = int(delay), beta=beta, l=1)
            bo.run(T, REGRET_PATH)
