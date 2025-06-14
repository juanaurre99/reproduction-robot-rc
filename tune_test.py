import numpy as np
from rc import ReservoirComputing
from tuner import Tuner
import matplotlib.pyplot as plt

# Helper to build a random sparse reservoir
def random_sparse_reservoir(n, spectral_radius=1.0, sparsity=0.1, seed=None):
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    num_nonzero = int(np.round(n * sparsity))
    for i in range(n):
        idx = rng.choice(n, size=num_nonzero, replace=False)
        W[i, idx] = rng.uniform(-1, 1, size=num_nonzero)
    eigs = np.linalg.eigvals(W)
    W *= spectral_radius / np.max(np.abs(eigs))
    return W

def random_input_layer(n, input_dim, scale=1.0, seed=None):
    rng = np.random.default_rng(seed)
    return scale * rng.uniform(-1, 1, (n, input_dim))

# Sinewave data
T = 500
t = np.linspace(0, 10 * np.pi, T)
y = np.sin(t)[np.newaxis, :]      # shape (1, T)
x = y[:, :-1]
y_target = y[:, 1:]

train_len = 400
x_train = x[:, :train_len]
y_train = y_target[:, :train_len]
x_val = x[:, train_len:]
y_val = y_target[:, train_len:]

input_dim = 1
output_dim = 1
n = 100

def ensure_column_major(x, target_dim):
    return x if x.shape[0] == target_dim else x.T
x_train = ensure_column_major(x_train, input_dim)
y_train = ensure_column_major(y_train, output_dim)
x_val   = ensure_column_major(x_val, input_dim)
y_val   = ensure_column_major(y_val, output_dim)

# Debug prints for initial shapes
print("DEBUG: x_train.shape =", x_train.shape)
print("DEBUG: y_train.shape =", y_train.shape)
print("DEBUG: x_val.shape   =", x_val.shape)
print("DEBUG: y_val.shape   =", y_val.shape)

# Build initial layers
reservoir_layer = random_sparse_reservoir(n, spectral_radius=1.0, sparsity=0.1, seed=42)
input_layer = random_input_layer(n, input_dim, scale=1.0, seed=42)
rc = ReservoirComputing(reservoir_layer, input_layer, washup_length=10, ridge_beta=1e-6)

print("DEBUG: rc.input_layer.shape =", rc.input_layer.shape)
print("DEBUG: rc.reservoir_layer.shape =", rc.reservoir_layer.shape)

search_space = {
    'alpha':           ("float", 0.2, 1.0),
    'kb':              ("float", -2.0, 2.0),
    'ridge_beta':      ("float", 1e-8, 1e-2, "log"),
    'spectral_radius': ("float", 0.7, 1.5),
    'input_scaling':   ("float", 0.1, 2.0),
}

def func_rmse(a, b, time_start, time_end):
    a = np.asarray(a)
    b = np.asarray(b)
    len_a = max(a.shape)
    if a.shape[1] != len_a:
        a = a.T
    if b.shape[1] != len_a:
        b = b.T
    a_window = a[:, time_start:time_end+1]
    b_window = b[:, time_start:time_end+1]
    rmse = np.sqrt(np.mean(np.sum((a_window - b_window) ** 2, axis=0)))
    return rmse

# Helper to run a sequence through your RC one-step-at-a-time
def predict_sequence(model, x_seq):
    T = x_seq.shape[1]
    model.reset_state()
    y_preds = []
    for ti in range(T):
        u = x_seq[:, ti]
        y = model.predict(u)
        y_preds.append(y)
    y_preds = np.stack(y_preds, axis=1)
    return y_preds

def sine_evaluator(model, trial):
    print("\nDEBUG: In sine_evaluator")
    print("DEBUG: x_val.shape =", x_val.shape)
    print("DEBUG: Model input_layer.shape =", model.input_layer.shape)
    print("DEBUG: First column x_val[:,0].shape =", x_val[:,0].shape)
    y_pred = predict_sequence(model, x_val)
    print("DEBUG: y_pred.shape =", y_pred.shape)
    print("DEBUG: y_val.shape =", y_val.shape)
    return func_rmse(y_pred, y_val, 0, y_val.shape[1]-1)

# Run tuner
tuner = Tuner(
    model=rc,
    search_space=search_space,
    x_train=x_train, y_train=y_train,
    evaluator=sine_evaluator,
    n_trials=5,
    verbose=True
)

tuner.optimize()
tuner.report()

# Plot predictions with best hyperparams
best_params = tuner.study.best_trial.params
rc.set_hp(**best_params)
rc.fit(x_train, y_train)
y_pred = predict_sequence(rc, x_val)

plt.figure(figsize=(8, 4))
plt.plot(y_val.flatten(), label="True")
plt.plot(y_pred.flatten(), '--', label="Predicted (best RC)")
plt.legend()
plt.title("Reservoir Computing Sine Prediction (Validation)")
plt.show()
