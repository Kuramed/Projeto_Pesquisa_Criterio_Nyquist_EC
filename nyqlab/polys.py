"""Utilitários de polinômios implementados do zero (apenas NumPy básico).

Convenção: coeficientes em potências DESCENDENTES de s (igual a np.polyval).
"""
from math import factorial

import numpy as np


def trim(p, tol=0.0):
    """Remove zeros à esquerda (coeficientes líderes nulos)."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    nz = np.flatnonzero(np.abs(p) > tol)
    return p[nz[0]:] if nz.size else np.array([0.0])


def poly_roots(p, init=None, tol=1e-14, max_iter=500):
    """Raízes de um polinômio pelo método de Aberth-Ehrlich (sem np.roots).

    init : chutes iniciais (opcional). No Lugar das Raízes, usa-se a solução do
           ganho anterior (continuação), o que mantém os ramos contínuos.
    Zeros exatos na origem (coeficientes finais nulos) são extraídos antes.
    """
    p = trim(p)
    k = 0
    while len(p) > 1 and p[-1] == 0.0:
        p = p[:-1]
        k += 1
    zeros0 = np.zeros(k, dtype=complex)
    n = len(p) - 1
    if n <= 0:
        return zeros0
    p = p / p[0]
    dp = np.polyder(p)

    if init is not None and len(init) == n + k:
        # descarta os k chutes mais próximos de zero (raízes na origem extraídas)
        init = np.asarray(init, dtype=complex)
        if k:
            idx = np.argsort(np.abs(init))[k:]
            init = init[idx]
        z = init.copy() + 1e-9 * (1 + 1j)
    else:
        r = max(abs(p[-1]) ** (1.0 / n), 1e-3)           # média geométrica dos módulos
        ang = 2 * np.pi * np.arange(n) / n + 0.4
        z = r * np.exp(1j * ang)

    for _ in range(max_iter):
        pz = np.polyval(p, z)
        dpz = np.polyval(dp, z)
        dpz = np.where(dpz == 0, 1e-300, dpz)
        ratio = pz / dpz
        diff = z[:, None] - z[None, :]
        np.fill_diagonal(diff, np.inf)
        s = np.sum(1.0 / diff, axis=1)
        w = ratio / (1.0 - ratio * s)
        z = z - w
        if np.max(np.abs(w) / (1.0 + np.abs(z))) < tol:
            break

    # polimento por Newton
    for _ in range(3):
        dpz = np.polyval(dp, z)
        dpz = np.where(dpz == 0, 1e-300, dpz)
        z = z - np.polyval(p, z) / dpz

    # limpeza: raízes reais têm parte imaginária numericamente nula
    z = np.where(np.abs(z.imag) < 1e-9 * (1 + np.abs(z.real)), z.real + 0j, z)
    return np.concatenate([zeros0, z])


def pade(tau, n):
    """Aproximação de Padé de ordem n para e^{-tau s}. Retorna (num, den)."""
    c = [factorial(2 * n - k) * factorial(n) /
         (factorial(2 * n) * factorial(k) * factorial(n - k)) for k in range(n + 1)]
    num = np.array([c[k] * (-tau) ** k for k in range(n + 1)])[::-1]
    den = np.array([c[k] * (tau) ** k for k in range(n + 1)])[::-1]
    return num, den


def poly_from_roots(roots, gain=1.0):
    """Polinômio mônico (vezes gain) a partir das raízes (convolução)."""
    p = np.array([1.0 + 0j])
    for r in roots:
        p = np.convolve(p, [1.0, -r])
    return (gain * p).real
