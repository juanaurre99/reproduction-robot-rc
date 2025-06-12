import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from robot import TwoLinkRobot
from rc import ReservoirComputing

# ---- 1. Load all data and parameters from MATLAB files ----

main_matfile = 'all_traj_06112025_341_7413.mat'  # Change as needed
params = scipy.io.loadmat(main_matfile, struct_as_record=False, squeeze_me=True)

def mat_struct_to_dict(mat_struct):
    if hasattr(mat_struct, '_fieldnames'):
        return {field: getattr(mat_struct, field) for field in mat_struct._fieldnames}
    elif isinstance(mat_struct, dict):
        return mat_struct
    elif isinstance(mat_struct, np.void):
        return {field: mat_struct[field] for field in mat_struct.dtype.names}
    return mat_struct

# --- Extract everything ---
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

# ---- SANITY CHECKS: RC layer shapes ----
if res_net.shape[0] != res_net.shape[1]:
    raise ValueError(f"res_net (reservoir_layer) must be square! Got {res_net.shape}")
if W_in.shape[0] != res_net.shape[0]:
    raise ValueError(f"W_in shape mismatch: rows {W_in.shape[0]} != reservoir size {res_net.shape[0]}")
if Wout is not None and Wout.shape[1] != res_net.shape[0]:
    raise ValueError(f"Wout shape mismatch: cols {Wout.shape[1]} != reservoir size {res_net.shape[0]}")
if not all([np.isscalar(x) for x in [alpha, kb, n, ridge_beta]]):
    raise ValueError(f"RC parameter type mismatch!")

# ---- Validation data (trajectory) ----
traj = scipy.io.loadmat('xy_val_traj.mat', struct_as_record=False, squeeze_me=True)
x_control = traj['x_control'].squeeze()
y_control = traj['y_control'].squeeze()
qdt_control = traj['qdt_control']
q_control = traj['q_control']
val_length = len(x_control)
data_control = np.column_stack((x_control, y_control))

# ---- Training data (from your .mat) ----
## Load .mat data
data = scipy.io.loadmat('fixed_robot_training_data.mat', struct_as_record=False, squeeze_me=True)

xy = data['xy']       # shape: (N, 2)
qdt = data['qdt']     # shape: (N, 2)
tau = data['tau']     # shape: (N, 2)

N = xy.shape[0] - 1   # so t+1 is in bounds

# Build train_x and train_y for the whole data
train_x = np.hstack([
    xy[0:N, :],      # x_t, y_t
    xy[1:N+1, :],    # x_{t+1}, y_{t+1}
    qdt[0:N, :],     # qd1_t, qd2_t
    qdt[1:N+1, :]    # qd1_{t+1}, qd2_{t+1}
])                  # shape: (N, 8)

train_y = tau[0:N, :]  # shape: (N, 2)

# Transpose so shapes are [8, N], [2, N] (matching ReservoirComputing convention)
train_x = train_x.T
train_y = train_y.T

print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
# --------- 2. Create Robot and RC ---------
robot = TwoLinkRobot(m1, m2, l1, l2, lc1, lc2, I1, I2, dt=dt)

# --- SANITY CHECK: Wout loaded? ---
if Wout is None or (hasattr(Wout, 'size') and Wout.size == 0):
    raise RuntimeError("Wout (output_layer) is missing or empty! Reservoir controller is untrained.")

rc = ReservoirComputing(
    res_net, W_in, output_layer=Wout,
    alpha=alpha, kb=kb,
    washup_length=washup_length,
    ridge_beta=ridge_beta,
    r_zero=None
)

# --------- 3. Train RC (fit) ---------
rc.fit(train_x, train_y)

# --------- 4. Run Closed-loop Validation ---------
disturbance_failure = np.zeros((2, val_length))
measurement_failure = np.zeros((4, val_length))
taudt_threshold = np.array([-5e-2, 5e-2])

q_init = q_control[0, :]
qdt_init = qdt_control[0, :]

def run_closed_loop_control(
        rc, robot: 'TwoLinkRobot',
        dt, val_length, data_control, qdt_control,
        disturbance_failure, measurement_failure, taudt_threshold,
        q_init, qdt_init
):
    # --- SANITY CHECK: output_layer present? ---
    if rc.output_layer is None:
        raise RuntimeError("ReservoirComputing output_layer is None. The controller must be trained or weights loaded!")

    n = rc.n
    q_pred = np.zeros((val_length + 100, 2))
    qdt_pred = np.zeros((val_length + 100, 2))
    q2dt_pred = np.zeros((val_length + 100, 2))
    tau_pred = np.zeros((val_length + 100, 2))
    data_pred = np.zeros((val_length + 100, 2))
    u = np.zeros(8)
    r = rc.r_zero.copy()
    q_pred[0, :] = q_init
    qdt_pred[0, :] = qdt_init
    u = np.hstack([
        data_control[0, :],  # x1, y1
        data_control[1, :],  # x2, y2
        qdt_control[0, :],   # qd1_1, qd2_1
        qdt_control[1, :]    # qd1_2, qd2_2
    ])
    print("Initial values:")
    print("u =", u)
    print("r =", r)
    print("q_pred[0] =", q_pred[0, :])
    print("qdt_pred[0] =", qdt_pred[0, :])

    for t_i in range(val_length - 3):
        predict_value, r = rc.step(u, r)
        # --- SANITY CHECK: None output from step ---
        if predict_value is None:
            raise RuntimeError(
                f"ReservoirComputing.step() returned None at t={t_i}! "
                f"This usually means output_layer is not set or has incorrect shape. "
                f"u shape: {u.shape}, r shape: {r.shape}, output_layer shape: {rc.output_layer.shape if rc.output_layer is not None else None}"
            )
        # --- SANITY CHECK: predict_value shape ---
        if predict_value.shape != (2,):
            raise ValueError(f"predict_value has unexpected shape {predict_value.shape}, expected (2,) at t={t_i}")

        predict_value = predict_value.copy()
        predict_value += predict_value * disturbance_failure[:, t_i]
        time_li = 0 if t_i == 0 else t_i - 1
        for li in range(2):
            diff = predict_value[li] - tau_pred[time_li, li]
            if diff > taudt_threshold[1] * dt:
                predict_value[li] = tau_pred[time_li, li] + taudt_threshold[1] * dt
            if diff < taudt_threshold[0] * dt:
                predict_value[li] = tau_pred[time_li, li] + taudt_threshold[0] * dt
        tau_pred[t_i, :] = predict_value
        time_now = t_i

        q2dt_pred[time_now, :] = robot.forward_dynamics(
            q_pred[time_now, :], qdt_pred[time_now, :], predict_value
        )

        q_pred[time_now + 1, :] = q_pred[time_now, :] + qdt_pred[time_now, :] * dt
        qdt_pred[time_now + 1, :] = qdt_pred[time_now, :] + q2dt_pred[time_now, :] * dt
        x_pred, y_pred = robot.forward_kinematics(q_pred[time_now + 1, 0], q_pred[time_now + 1, 1])
        x_pred_measurement = x_pred + x_pred * measurement_failure[0, t_i]
        y_pred_measurement = y_pred + y_pred * measurement_failure[1, t_i]
        qdt_measurement_f_value = qdt_pred[time_now + 1, :] * measurement_failure[2:4, t_i]
        qdt_pred_measurement = qdt_pred[time_now + 1, :] + qdt_measurement_f_value
        data_pred[time_now + 1, :] = [x_pred, y_pred]
        u[0:2] = [x_pred_measurement, y_pred_measurement]
        u[2:4] = data_control[time_now + 2, :]
        u[4:6] = qdt_pred_measurement
        u[6:8] = qdt_control[time_now + 2, :]
        if t_i == 0:
            print("After first iteration:")
            print("r =", r)
            print("predict_value =", predict_value)
            print("q_pred[1] =", q_pred[1, :])
            print("qdt_pred[1] =", qdt_pred[1, :])
        if t_i == val_length - 4:
            print("Final values:")
            print("r =", r)
            print("predict_value =", predict_value)
            print("q_pred[{}] =".format(time_now + 1), q_pred[time_now + 1, :])
            print("qdt_pred[{}] =".format(time_now + 1), qdt_pred[time_now + 1, :])

    return data_pred[:val_length, :]

data_pred = run_closed_loop_control(
    rc=rc,
    robot=robot,
    dt=dt, val_length=val_length,
    data_control=data_control,
    qdt_control=qdt_control,
    disturbance_failure=disturbance_failure,
    measurement_failure=measurement_failure,
    taudt_threshold=taudt_threshold,
    q_init=q_init,
    qdt_init=qdt_init
)

# --------- 6. Plot results ---------
plt.figure(figsize=(7, 6))
plt.plot(data_control[:, 0], data_control[:, 1], label='Ground Truth', linewidth=2)
plt.plot(data_pred[:, 0], data_pred[:, 1], '--', label='Reservoir Control', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Reservoir Closed-Loop Control Trajectory')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()
