import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


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

def load_reservoir_parameters(mat_path):
    def mat_struct_to_dict(mat_struct):
        if hasattr(mat_struct, '_fieldnames'):
            return {field: getattr(mat_struct, field) for field in mat_struct._fieldnames}
        elif isinstance(mat_struct, dict):
            return mat_struct
        elif isinstance(mat_struct, np.void):
            return {field: mat_struct[field] for field in mat_struct.dtype.names}
        return mat_struct

    params = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    time_info = mat_struct_to_dict(params['time_infor'])
    washup_length = int(time_info['washup_length'])
    val_length = int(time_info['val_length'])
    dt = float(params['dt'])

    res_info = mat_struct_to_dict(params['res_infor'])
    W_in = res_info['W_in']
    res_net = res_info['res_net']
    alpha = float(res_info['alpha'])
    kb = float(res_info['kb'])
    ridge_beta = float(res_info['beta'])
    Wout = params['Wout']

    properties = params['properties']
    m1, m2, l1, l2, lc1, lc2, I1, I2 = properties

    return {
        'washup_length': washup_length,
        'val_length': val_length,
        'dt': dt,
        'W_in': W_in,
        'res_net': res_net,
        'alpha': alpha,
        'kb': kb,
        'ridge_beta': ridge_beta,
        'Wout': Wout,
        'properties': {
            'm1': m1, 'm2': m2, 'l1': l1, 'l2': l2,
            'lc1': lc1, 'lc2': lc2, 'I1': I1, 'I2': I2
        }
    }

def plot_trajectory_comparison(ground_truth, prediction, save_path="trajectory_plot.png", title="Closed-Loop Reservoir Control"):

    plt.figure(figsize=(6, 6))
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', linewidth=2)
    plt.plot(prediction[:, 0], prediction[:, 1], '--', label='RC Prediction', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🖼️ Saved static plot: {save_path}")

