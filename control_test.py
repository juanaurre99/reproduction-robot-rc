import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from control import *
from robot import TwoLinkRobot
from rc import ReservoirComputing
from utils import *

# ----------- 1. Load Data and Params from MATLAB files -----------
main_matfile = 'all_traj_06112025_341_7413.mat'  # Adjust path as needed
params = scipy.io.loadmat(main_matfile, struct_as_record=False, squeeze_me=True)


traj_name= "lorenz"

def mat_struct_to_dict(mat_struct):
    if hasattr(mat_struct, '_fieldnames'):
        return {field: getattr(mat_struct, field) for field in mat_struct._fieldnames}
    elif isinstance(mat_struct, dict):
        return mat_struct
    elif isinstance(mat_struct, np.void):
        return {field: mat_struct[field] for field in mat_struct.dtype.names}
    return mat_struct

# --- Extract RC/robot/trajectory data
time_infor = mat_struct_to_dict(params['time_infor'])
washup_length = int(time_infor['washup_length'])
val_length = int(time_infor['val_length'])
dt = float(params['dt'])

res_info = params['res_infor']
if isinstance(res_info, np.ndarray): res_info = res_info.item()
res_info = mat_struct_to_dict(res_info)

W_in = res_info['W_in']
res_net = res_info['res_net']
alpha = float(res_info['alpha'])
kb = float(res_info['kb'])
n = int(res_info['n'])
ridge_beta = float(res_info['beta'])
Wout = params['Wout']

properties = params['properties']
m1, m2, l1, l2, lc1, lc2, I1, I2 = properties

trajectory, q_control, qdt_control, val_length, x_control, y_control = load_validation_trajectory(traj_name)


# ---- Training data (from .mat) ----
data = scipy.io.loadmat('fixed_robot_training_data.mat', struct_as_record=False, squeeze_me=True)
xy = data['xy']       # (N, 2)
qdt = data['qdt']     # (N, 2)
tau = data['tau']     # (N, 2)
N = xy.shape[0] - 1

train_x = np.hstack([
    xy[0:N, :],      # x_t, y_t
    xy[1:N+1, :],    # x_{t+1}, y_{t+1}
    qdt[0:N, :],     # qd1_t, qd2_t
    qdt[1:N+1, :]    # qd1_{t+1}, qd2_{t+1}
])
train_y = tau[0:N, :]
train_x = train_x.T
train_y = train_y.T

print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)

# ----------- 2. Define the robot and RC -----------
robot = TwoLinkRobot(m1, m2, l1, l2, lc1, lc2, I1, I2, dt=dt)
rc = ReservoirComputing(
    res_net, W_in, output_layer=Wout,
    alpha=alpha, kb=kb,
    washup_length=washup_length,
    ridge_beta=ridge_beta,
    r_zero=None
)

# ----------- 3. Train RC (if not loaded) -----------
#rc.fit(train_x, train_y)

# ----------- 4. Define the control_loop function (paste from above or import) -----------

# (Paste the control_loop function here or import it)

# ----------- 5. Run closed-loop control -----------
taudt_threshold = np.array([-5e-2, 5e-2])
disturbance_failure = np.zeros((2, val_length))
measurement_failure = np.zeros((4, val_length))

data_pred = control_loop(
    rc=rc,
    robot=robot,
    trajectory=trajectory,
    qdt_traj=qdt_control,
    taudt_threshold=taudt_threshold,
    disturbance_failure=disturbance_failure,
    measurement_failure=measurement_failure,
    q_init=q_control[0, :],        # <--- actual initial joint config
    qdt_init=qdt_control[0, :],    # <--- actual initial joint velocity
    verbose=True
)


# ----------- 6. Plot results -----------
plt.figure(figsize=(7, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Ground Truth', linewidth=2)
plt.plot(data_pred[:, 0], data_pred[:, 1], '--', label='Reservoir Control', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Reservoir Closed-Loop Control Trajectory')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()
