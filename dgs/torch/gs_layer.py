# gs_layer.py
# Differentiable Gerchberg–Saxton layer with dispersion.

import torch
import torch.nn as nn
import torch.fft as fft

class DispersionForward(nn.Module):
    def __init__(self, beta: torch.Tensor):
        super().__init__()
        self.register_buffer("beta", beta)

    def forward(self, x):
        X = fft.fft(x, dim=-1)
        D = torch.exp(1j * self.beta)
        Y = X * D
        y_time = fft.ifft(Y, dim=-1)
        return torch.abs(y_time) ** 2


class GSLayer(nn.Module):
    def __init__(self, beta, measured_intensity, n_iters=1):
        super().__init__()
        self.forward_op = DispersionForward(beta)
        self.register_buffer("meas_int", measured_intensity)
        self.n_iters = n_iters

    def forward(self, x0):
        x = x0
        for _ in range(self.n_iters):
            # Forward intensity
            y = self.forward_op(x)

            # Amplitude constraint
            amp_meas = torch.sqrt(self.meas_int + 1e-12)
            amp_est  = torch.sqrt(y + 1e-12)
            scale = amp_meas / (amp_est + 1e-12)

            # Recompute field to extract phase
            X = fft.fft(x, dim=-1)
            D = torch.exp(1j * self.forward_op.beta)
            field = fft.ifft(X * D, dim=-1)
            phase = torch.angle(field)

            # Update estimate
            x = amp_meas * torch.exp(1j * phase)

        return x
