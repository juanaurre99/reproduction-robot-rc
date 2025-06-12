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

    def compute_reservoir_state(self, u, r=None):
        if r is None:
            r = self.r_zero.copy()
        r_next = (1 - self.alpha) * r + self.alpha * np.tanh(
            self.reservoir_layer @ r + self.input_layer @ u + self.kb * np.ones(self.n)
        )
        return r_next

    def _reservoir_output_transform(self, r):
        r_out = r.copy()
        r_out[1::2] = r_out[1::2] ** 2
        return r_out

    def step(self, u, r=None):
        if r is None:
            r = self.r_zero.copy()
        r_next = self.compute_reservoir_state(u, r)
        r_out = self._reservoir_output_transform(r_next)
        y = None if self.output_layer is None else self.output_layer @ r_out
        return y, r_next

    def fit(self, train_x, train_y):
        n, train_length = self.n, train_x.shape[1]
        wash = self.washup_length
        beta = self.ridge_beta
        r_all = np.zeros((n, train_length + 1))
        for ti in range(train_length):
            r_all[:, ti + 1] = self.compute_reservoir_state(train_x[:, ti], r_all[:, ti])
        r_out = r_all[:, wash + 1:]  # (n, train_length - washup)
        r_out = self._reservoir_output_transform(r_out)
        y_train = train_y[:, wash:]  # (output_dim, train_length - washup)
        RR = r_out @ r_out.T + beta * np.eye(n)
        self.output_layer = y_train @ r_out.T @ np.linalg.inv(RR)
        self.r_zero = r_all[:, -1].copy()
        return self.output_layer
