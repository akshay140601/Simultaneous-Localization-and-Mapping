'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    '''print(last_X[0])
    print(X[0])
    print(last_X[1])
    print(X[1])'''
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    if -np.pi<=angle_rad<=np.pi:
        angle_rad_pass = angle_rad
    elif angle_rad > np.pi:
        domain = 2 * np.pi
        x = angle_rad // domain
        angle_rad_new = angle_rad - (x * 2 * np.pi)
        if angle_rad_new < np.pi:
            angle_rad_pass = angle_rad_new
        else:
            angle_rad_pass = angle_rad_new - (2 * np.pi)
    else:
        domain = -(2 * np.pi)
        x = angle_rad // domain
        angle_rad_new = angle_rad + (x * 2 * np.pi)
        if angle_rad_new > -np.pi:
            angle_rad_pass = angle_rad_new
        else:
            angle_rad_pass = angle_rad_new + (2 * np.pi)

    return angle_rad_pass


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    #landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))
    landmark = []
    #landmark_cov = []

    x, y, theta = init_pose[0], init_pose[1], init_pose[2]
    j = 0
    combined_cov = np.block([[init_pose_cov, np.zeros((3, 2))],
                             [np.zeros((2, 3)), init_measure_cov]])
    for i in range(k):
        beta = init_measure[j]
        r = init_measure[j + 1]
        lx = x + r * np.cos(theta + beta)
        ly = y + r * np.sin(theta + beta)
        landmark.append(lx)
        landmark.append(ly)

        J_1_1 = 1
        J_1_2 = 0
        J_1_3 = -r * np.sin(theta + beta)
        J_1_4 = -np.sin(theta + beta)
        J_1_5 = np.cos(theta + beta)
        J_2_1 = 0
        J_2_2 = 1
        J_2_3 = r * np.cos(theta + beta)
        J_2_4 = np.cos(theta + beta)
        J_2_5 = np.sin(theta + beta)
        J = np.block([[J_1_1, J_1_2, J_1_3, J_1_4, J_1_5], [J_2_1, J_2_2, J_2_3, J_2_4, J_2_5]])
        #landmark_cov[j:j+2, j:j+2] = (J @ init_pose_cov @ (J.T)) + (init_measure_cov)
        landmark_cov[j:j+2, j:j+2] = J @ combined_cov @ (J.T)

        j += 2
    landmark = np.array(landmark).reshape((-1, 1))
    #print(landmark_cov)
    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''

    dt = control[0]
    alpha = control[1]
    x, y, theta = X[0], X[1], X[2]
    xt1 = x + dt * np.cos(theta)
    yt1 = y + dt * np.sin(theta)
    thetat1 = theta + alpha
    X_pre = np.zeros((3 + 2*k, 1))
    X_pre[0] = xt1
    X_pre[1] = yt1
    X_pre[2] = thetat1
    X_pre[3:3+2*k] = X[3:3+2*k]
    Gt = np.eye(3 + 2*k)
    Gt[0:3, 0:3] = np.block([[1, 0, (-dt * np.sin(theta))],
                  [0, 1, (dt * np.cos(theta))],
                  [0, 0, 1]])
    transformation_matrix = np.block([[np.cos(theta), -np.sin(theta), 0],
                                      [np.sin(theta), np.cos(theta), 0],
                                      [0, 0, 1]])
    control_cov_world_frame = transformation_matrix @ control_cov @ (transformation_matrix.T)
    control_cov_updated = np.block([[control_cov_world_frame, np.zeros((3, 2*k))],
                                    [np.zeros((2*k, 3)), np.zeros((2*k, 2*k))]])
    P_pre = (Gt @ P @ (Gt.T)) + control_cov_updated
    #print(X_pre)

    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    # Calculating measurement Jacobian wrt the state vector (Ht)
    Ht = np.zeros((2*k, 3 + 2*k))
    # Defining a matrix for measurement covariance
    Qt = np.zeros((2*k, 2*k))
    # Defining a matrix to store the predicted beta and r values.
    # This is used to calculate the innovation (actual measurement - predicted measurement)
    pred_meas = np.zeros((2*k, 1))
    x, y, theta = X_pre[0], X_pre[1], X_pre[2]
    j = 0
    for i in range(k):
        lx = X_pre[3 + j]
        ly = X_pre[3 + j+1]
        eu_dist = (lx - x)**2 + (ly - y)**2
        Ht[j, 0] = (ly - y) / eu_dist
        Ht[j, 1] = (x - lx) / eu_dist
        Ht[j, 2] = -1
        Ht[j, 3 + j] = (y - ly) / eu_dist
        Ht[j, 3 + j+1] = (lx - x) / eu_dist
        Ht[j+1, 0] = (x - lx) / np.sqrt(eu_dist)
        Ht[j+1, 1] = (y - ly) / np.sqrt(eu_dist)
        Ht[j+1, 2] = 0
        Ht[j+1, 3 + j] = (lx - x) / np.sqrt(eu_dist)
        Ht[j+1, 3 + j+1] = (ly - y) / np.sqrt(eu_dist)
        pred_meas[j] = warp2pi(np.arctan2(ly - y, lx - x) - theta)
        pred_meas[j+1] = np.sqrt(eu_dist)
        Qt[j:j+2, j:j+2] = measure_cov

        j += 2
    
    # Calculating Kalman Gain
    Kt = P_pre @ (Ht.T) @ np.linalg.inv((Ht @ P_pre @ (Ht.T) + Qt))
    # Correcting the state vector
    #measure_updated = np.append([0, 0, 0], measure).reshape(-1, 1)
    #pred_meas_updated = np.append([0, 0, 0], pred_meas).reshape(-1, 1)
    X = X_pre + Kt @ (measure - pred_meas)
    P = (np.eye(3 + 2*k) - Kt @ Ht) @ P_pre
    #print(Ht)
    #print(P_pre)

    return X, P


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    j = 0
    for i in range(k):
        lx_pred, ly_pred = X[3 + j], X[3 + j+1]
        euclidean_distance = np.sqrt((l_true[j] - lx_pred)**2 + (l_true[j+1] - ly_pred)**2)
        print(f"The Euclidean distance for landmark {i+1} is: ", euclidean_distance.item())
        cov_matrix = P[3 + j: 3 + j+2, 3 + j: 3 + j+2]
        x_minus_mean = np.array([l_true[j] - lx_pred, l_true[j+1] - ly_pred]).reshape(1, -1)
        mahalanobis_distance = np.sqrt(x_minus_mean @ np.linalg.inv(cov_matrix) @ (x_minus_mean.T))
        print(f"The Mahalanobis distance for landmark {i+1} is: ", mahalanobis_distance.item())

        j += 2
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25;
    sig_y = 0.1;
    sig_alpha = 0.1;
    sig_beta = 0.01/1000;
    sig_r = 0.08/1000;


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    #print(last_X)
    draw_traj_and_map(X, last_X, P, 0)
    #print(last_X)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])
        #print(last_X)

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            #print(last_X)
            X_pre, P_pre = predict(X, P, control, control_cov, k)
            #print(last_X)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)
            #print(X)
            #print(last_X)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    #print("The final covariance matrix")
    #print(P)
    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
