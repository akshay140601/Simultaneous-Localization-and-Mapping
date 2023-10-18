'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import random

class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        # Number of particles
        self._num_particles = 500


        # Adaptive no of particles



    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """

        
        X_bar_resampled = []
        M = 1 / self._num_particles
        X_bar[:, 3] = X_bar[:, 3] / np.sum(X_bar[:, 3])
        r = random.uniform(0, M)
        c = X_bar[0, 3]
        i = 0
        for m in range(1, self._num_particles + 1):
            U = r + (m - 1) * M
            while (U > c):
                i = i + 1
                c = c + X_bar[i, 3]
            X_bar_resampled.append(X_bar[i])
        X_bar_resampled_np = np.array(X_bar_resampled)
        #X_bar_resampled_np_array = X_bar_resampled_np.reshape(self._num_particles, 4)
        #X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled_np
