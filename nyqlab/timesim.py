"""Simulação no tempo (RK4) de malha fechada com atraso puro na realimentação."""
import numpy as np

from .polys import trim


def tf2ss(num, den):
    """Forma canônica controlável. Retorna A, B, C, D (G própria)."""
    num, den = trim(num), trim(den)
    num = num / den[0]
    den = den / den[0]
    n = len(den) - 1
    b = np.concatenate([np.zeros(n + 1 - len(num)), num])
    D = b[0]
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), D
    A = np.zeros((n, n))
    A[0, :] = -den[1:]
    A[1:, :-1] = np.eye(n - 1)
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    C = (b[1:] - D * den[1:]).reshape(1, n)
    return A, B, C, D


def _rk4(A, B, x, u, dt):
    f = lambda x_: A @ x_ + B[:, 0] * u
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_feedback(G, K=1.0, T=20.0, dt=1e-3, r=1.0, tau=None, blow=1e8):
    """Resposta ao degrau de r -> y com u = K (r - y(t - tau)).

    G : TF (usa só a parte racional; o atraso vem de `tau` ou, se None, de G.delay).
    Retorna (t, y, u); interrompe se |y| > blow (instabilidade).
    """
    tau = G.delay if tau is None else tau
    A, B, C, D = tf2ss(G.num, G.den)
    d = int(round(tau / dt))
    nsteps = int(round(T / dt))
    x = np.zeros(A.shape[0])
    t = np.arange(nsteps) * dt
    y = np.zeros(nsteps)
    u = np.zeros(nsteps)
    for k in range(nsteps):
        if d > 0:
            yd = y[k - d] if k >= d else 0.0
            uk = K * (r - yd)
        else:
            uk = K * (r - (C @ x).item() if x.size else r) / (1.0 + K * D) if D else K * (r - (C @ x).item())
        u[k] = uk
        y[k] = (C @ x).item() + D * uk if x.size else D * uk
        if abs(y[k]) > blow or not np.isfinite(y[k]):
            return t[:k + 1], y[:k + 1], u[:k + 1]
        if x.size:
            x = _rk4(A, B, x, uk, dt)
    return t, y, u
