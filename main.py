import numpy as np
from control import control_loop
from robot import TwoLinkRobot
from rc import ReservoirComputing
from utils import (
    load_reservoir_parameters,
    load_validation_trajectory,
    func_rmse,
    plot_trajectory_comparison
)

params = load_reservoir_parameters("all_traj_06112025_341_7413.mat") # load parameters
val_length = params['val_length']
dt = params['dt']


traj_name = "lorenz"
trajectory, q_control, qdt_control, val_length, x_control, y_control = load_validation_trajectory(traj_name) #load trajectory


p = params['properties']
robot = TwoLinkRobot(p['m1'], p['m2'], p['l1'], p['l2'], p['lc1'], p['lc2'], p['I1'], p['I2'], dt=dt) # initialize robot

rc = ReservoirComputing(
    reservoir_layer=params['res_net'],
    input_layer=params['W_in'],
    output_layer=params['Wout'],
    alpha=params['alpha'],
    kb=params['kb'],
    washup_length=params['washup_length'],
    ridge_beta=params['ridge_beta']
)


data_pred, q_pred = control_loop(
    rc=rc,
    robot=robot,
    trajectory=trajectory,
    qdt_traj=qdt_control,
    taudt_threshold=np.array([-5e-2, 5e-2]),
    disturbance_failure=np.zeros((2, val_length)),
    measurement_failure=np.zeros((4, val_length)),
    q_init=q_control[0, :],
    qdt_init=qdt_control[0, :],
    verbose=False
)


rmse = func_rmse(trajectory.T, data_pred.T, 0, val_length - 1)
print(f"📉 Tracking RMSE: {rmse:.5f}")


plot_trajectory_comparison(trajectory, data_pred)
