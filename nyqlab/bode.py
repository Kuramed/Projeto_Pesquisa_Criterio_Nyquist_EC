"""Diagrama de Bode: resposta exata e aproximação assintótica de módulo."""
import numpy as np

from .nyquist import margins


def bode_data(G, K=1.0, w=None):
    """Retorna (w, mag_dB, fase_graus) com fase desenrolada (atraso incluso exatamente)."""
    if w is None:
        w = np.geomspace(1e-3, 1e3, 4000)
    L = K * G.freq(w)
    return w, 20 * np.log10(np.abs(L)), np.degrees(np.unwrap(np.angle(L)))


def asymptotic_magnitude(G, w, K=1.0):
    """Assíntotas de módulo (dB): cada raiz r ≠ 0 contribui com ±20 dB/déc a partir de |r|."""
    num, den = G.num.copy(), G.den.copy()
    m = 0                                              # (zeros - polos) na origem
    while len(num) > 1 and num[-1] == 0:
        num, m = num[:-1], m + 1
    while len(den) > 1 and den[-1] == 0:
        den, m = den[:-1], m - 1
    k0 = abs(K * num[-1] / den[-1])
    mag = 20 * np.log10(k0) * np.ones_like(w) + 20 * m * np.log10(w)
    from .polys import poly_roots
    for r in poly_roots(num):
        if abs(r) > 0:
            mag += 20 * np.log10(np.maximum(w / abs(r), 1.0))
    for r in poly_roots(den):
        if abs(r) > 0:
            mag -= 20 * np.log10(np.maximum(w / abs(r), 1.0))
    return mag


def bode_margins(G, K=1.0):
    """Margens lidas do Bode (mesmo algoritmo da resposta em frequência)."""
    return margins(G, K)


def plot_bode(axs, G, K=1.0, asymptotes=True):
    w, mag, ph = bode_data(G, K)
    axs[0].semilogx(w, mag, label="exato")
    if asymptotes:
        axs[0].semilogx(w, asymptotic_magnitude(G, w, K), "--", label="assintótico")
    axs[0].axhline(0, c="gray", lw=0.5)
    axs[0].set_ylabel("Módulo (dB)")
    axs[0].legend()
    axs[0].grid(alpha=0.3, which="both")
    axs[1].semilogx(w, ph)
    axs[1].axhline(-180, c="gray", lw=0.5)
    axs[1].set_ylabel("Fase (°)")
    axs[1].set_xlabel("ω (rad/s)")
    axs[1].grid(alpha=0.3, which="both")
