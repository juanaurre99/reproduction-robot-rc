import numpy as np

class ReservoirComputing:
    def __init__(self, reservoir_layer, input_layer, output_layer=None,
                 alpha=1.0, kb=0.0, r_zero=None,
                 washup_length=10, ridge_beta=1e-6):
        self.reservoir_layer = reservoir_layer
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.alpha = alpha
        self.kb = kb
        self.n = reservoir_layer.shape[0]
        self.r_zero = np.zeros(self.n) if r_zero is None else r_zero.copy()
        self.washup_length = washup_length
        self.ridge_beta = ridge_beta

        # New: Internal state
        self.reservoir_state = self.r_zero.copy()

    def reset_state(self, r_init=None):
        """Reset the internal reservoir state."""
        if r_init is None:
            self.reservoir_state = self.r_zero.copy()
        else:
            self.reservoir_state = r_init.copy()

    def compute_reservoir_state(self, u):
        r = self.reservoir_state
        r_next = (1 - self.alpha) * r + self.alpha * np.tanh(
            self.reservoir_layer @ r + self.input_layer @ u + self.kb * np.ones(self.n)
        )
        # Update internal state
        self.reservoir_state = r_next
        return r_next

    def _reservoir_output_transform(self, r):
        r_out = r.copy()
        r_out[1::2] = r_out[1::2] ** 2
        return r_out

    def predict(self, u):
        r_next = self.compute_reservoir_state(u)
        r_out = self._reservoir_output_transform(r_next)
        y = None if self.output_layer is None else self.output_layer @ r_out
        return y

    def fit(self, train_x, train_y):
        n, train_length = self.n, train_x.shape[1]
        wash = self.washup_length
        beta = self.ridge_beta
        r_all = np.zeros((n, train_length + 1))
        # TEMPORARILY reset state for training
        old_state = self.reservoir_state.copy()
        self.reservoir_state = self.r_zero.copy()
        for ti in range(train_length):
            r_all[:, ti + 1] = (
                (1 - self.alpha) * r_all[:, ti] +
                self.alpha * np.tanh(
                    self.reservoir_layer @ r_all[:, ti] +
                    self.input_layer @ train_x[:, ti] +
                    self.kb * np.ones(self.n)
                )
            )
        r_out = r_all[:, wash + 1:]  # (n, train_length - washup)
        r_out = self._reservoir_output_transform(r_out)
        y_train = train_y[:, wash:]  # (output_dim, train_length - washup)
        RR = r_out @ r_out.T + beta * np.eye(n)
        self.output_layer = y_train @ r_out.T @ np.linalg.inv(RR)
        self.r_zero = r_all[:, -1].copy()
        self.reservoir_state = self.r_zero.copy()
        # Restore state if needed (could also leave at reset)
        # self.reservoir_state = old_state
        return self.output_layer

    def set_hp(self, spectral_radius=None, input_scaling=None, alpha=None, kb=None, ridge_beta=None):
        if alpha is not None:
            self.alpha = alpha
        if kb is not None:
            self.kb = kb
        if ridge_beta is not None:
            self.ridge_beta = ridge_beta

        # Only rescale, don't rebuild matrix (sparsity/k not changed)
        if spectral_radius is not None:
            eigvals = np.linalg.eigvals(self.reservoir_layer)
            current_radius = np.max(np.abs(eigvals))
            if current_radius > 0:
                self.reservoir_layer *= spectral_radius / current_radius

        if input_scaling is not None:
            norm = np.linalg.norm(self.input_layer)
            if norm > 0:
                self.input_layer *= input_scaling / norm

        # Always reset state after changing structure/hyperparameters
        self.reservoir_state = self.r_zero.copy()
