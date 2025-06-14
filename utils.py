import os
import numpy as np
import scipy.io

def load_validation_trajectory(traj_name, folder="trajectories"):

    filename = os.path.join(folder, f"xy_val_traj_{traj_name}.mat")
    traj = scipy.io.loadmat(filename)
    x_control = traj['x_control'].squeeze()
    y_control = traj['y_control'].squeeze()
    qdt_control = traj['qdt_control']
    q_control = traj['q_control']
    val_length = len(x_control)
    trajectory = np.column_stack((x_control, y_control))
    qdt_traj = qdt_control
    return trajectory, q_control, qdt_control, val_length, x_control, y_control

import numpy as np

def func_rmse(a, b, time_start, time_end):

    a = np.asarray(a)
    b = np.asarray(b)

    # Make sure last dimension is 'time'
    len_a = max(a.shape)
    if a.shape[1] != len_a:
        a = a.T
    if b.shape[1] != len_a:
        b = b.T

    # Select the time window
    # MATLAB is 1-based and inclusive, Python is 0-based and [start:end+1]
    a_window = a[:, time_start:time_end+1]
    b_window = b[:, time_start:time_end+1]

    # Compute RMSE
    rmse = np.sqrt(np.mean(np.sum((a_window - b_window) ** 2, axis=0)))
    return rmse

