'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.00005
        self._alpha2 = 0.00005
        self._alpha3 = 0.0001
        self._alpha4 = 0.0001


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        #f_out = open("C:\Subjects\Subjects\SLAM\Assignments\log.txt", "a")
        #if u_t0[0] == u_t1[0] and u_t0[1] == u_t1[1] and u_t0[2] == u_t1[2]:
            #f_out.write('I am here'+"\n")
            #print("I'm here!!!!")
            #x = x_t0[0]
            #y = x_t0[1]
            #theta = x_t0[2]
            #f_out.write(str([x,y,theta])+"\n")
        #else:
        # Recover relative motion parameters from the odometry readings
        del_rot1 = math.atan2((u_t1[1] - u_t0[1]), (u_t1[0] - u_t0[0])) - u_t0[2]
        del_trans = math.sqrt((u_t0[0] - u_t1[0])**2 + (u_t0[1] - u_t1[1])**2)
        del_rot2 = u_t1[2] - u_t0[2] - del_rot1

        # Adding noise
        del_rot1_cap = del_rot1 - np.random.normal(0, np.sqrt(self._alpha1 * (del_rot1 * del_rot1) + self._alpha2 * (del_trans * del_trans)))
        del_trans_cap = del_trans - np.random.normal(0, np.sqrt(self._alpha3 * (del_trans * del_trans) + self._alpha4 * (del_rot1 * del_rot1) + self._alpha4 * (del_rot2 * del_rot2)))
        del_rot2_cap = del_rot2 - np.random.normal(0, np.sqrt(self._alpha1 * (del_rot2 * del_rot2) + self._alpha2 * (del_trans * del_trans)))

        # Computing particle state belief at time t
        x = x_t0[0] + del_trans_cap * np.cos(x_t0[2] + del_rot1_cap)
        y = x_t0[1] + del_trans_cap * np.sin(x_t0[2] + del_rot1_cap)
        theta = x_t0[2] + del_rot1_cap + del_rot2_cap
        '''
        f_out.write(str(u_t0)+"\n")
        f_out.write(str(u_t1)+"\n")
        f_out.write(str(x_t0)+"\n")
        f_out.write(str([x,y,theta])+"\n")
        print(u_t1)
        print(x_t0)
        print([x,y,theta])
        '''

        return [x,y,theta]
    
    '''if __name__=="__main__":
    motion = MotionModel()
    x = motion.update([-94.234001,-139.953995,-1.342158], [-94.234001,-139.953995,-1.340413], [5.21040954e+03,2.66512377e+02,-7.39994863e-02])
    print(x)'''
    '''
    [ 5.21040954e+03  2.66512377e+02 -7.39994863e-02] = xt0
    [ -94.234001 -139.953995   -1.342158] = ut0
    [ -94.234001 -139.953995   -1.342158] = ut1
    '''

