'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
import random


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])
    


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()
    


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


'''def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    freespace = []
    for i in range(occupancy_map.shape[0]):
        for j in range(occupancy_map.shape[1]):
            if occupancy_map[i][j] == 0:
                freespace.append([i, j])
            else:
                continue
    
    freespace_world_frame = []
    for a in range(len(freespace)):
        for b in range(5):
            freespace_world_frame.append([freespace[a][0]*10 + b, freespace[a][1]*10])
            freespace_world_frame.append([freespace[a][0]*10 - b, freespace[a][1]*10])
            freespace_world_frame.append([freespace[a][0]*10, freespace[a][1]*10 + b])
            freespace_world_frame.append([freespace[a][0]*10, freespace[a][1]*10 - b])

    x0_vals = []
    y0_vals = []
    x_and_y_vals = random.sample(freespace_world_frame, num_particles)
    for i in range(len(x_and_y_vals)):
        x0_vals.append(x_and_y_vals[i][0])
        y0_vals.append(x_and_y_vals[i][1])

    x0_vals_np = np.array(x0_vals)
    x0_vals_np = x0_vals_np.reshape(num_particles, 1)
    y0_vals_np = np.array(y0_vals)
    y0_vals_np = y0_vals_np.reshape(num_particles, 1)
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
                                         
    X_bar_init = np.hstack((y0_vals_np, x0_vals_np, theta0_vals, w0_vals))

    #X_bar_init = np.zeros((num_particles, 4))

    return X_bar_init'''
    
def init_particles_freespace(num_particles, occupancy_map):
    free_space_coords = np.argwhere(occupancy_map == 0)
    num_free_space_coords = free_space_coords.shape[0]

    random_indices = np.random.choice(num_free_space_coords, num_particles, replace=True)
    sampled_coords = free_space_coords[random_indices]

    x0_vals = sampled_coords[:, 1] * 10 + np.random.uniform(-5, 5, num_particles)
    y0_vals = sampled_coords[:, 0] * 10 + np.random.uniform(-5, 5, num_particles)
    theta0_vals = np.random.uniform(-3.14, 3.14, num_particles)

    w0_vals = np.ones(num_particles, dtype=np.float64) / num_particles

    X_bar_init = np.column_stack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init

def for_motion(num_particles):
    list = []
    for i in range(num_particles):
        list.append([4892, 916, 0.523599, 1])
    list_np = np.array(list)
    list_np_pass = list_np.reshape(num_particles, 4)
    return list_np_pass

if __name__ == '__main__':
    
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='data/map/wean.dat')
    parser.add_argument('--path_to_log', default='data/log/robotdata1.log')
    parser.add_argument('--output', default='results_motion')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)

    resampler = Resampling()

    num_particles = args.num_particles
    #X_bar = init_particles_random(num_particles, occupancy_map)
    #print(X_bar)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    #X_bar = for_motion(num_particles)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    #if args.visualize:
    visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            '''
            print(x_t0)
            print(u_t0)
            print(u_t1)
            '''
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            #print(x_t1)
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1, occupancy_map)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
                #print(X_bar_new)
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        #print(X_bar)
        u_t0 = u_t1

        """
        RESAMPLING
        """
        if (X_bar[0][3] != 0):
            X_bar = resampler.low_variance_sampler(X_bar)
        else:
            continue

        #args.visualize:
        
        #if time_idx%20==0:
            #visualize_map(occupancy_map)
        visualize_timestep(X_bar, time_idx, args.output)
