"""Critério de Routh-Hurwitz implementado do zero.

Casos especiais tratados:
  * zero na primeira coluna (linha não nula)  -> método do epsilon;
  * linha inteira de zeros                    -> polinômio auxiliar A(s) e sua derivada.
"""
from dataclasses import dataclass, field

import numpy as np

from .polys import trim, poly_roots


@dataclass
class RouthResult:
    table: np.ndarray
    first_column: np.ndarray
    sign_changes: int                 # = nº de raízes no semiplano direito
    status: str                       # 'estável' | 'marginal' | 'instável'
    special_cases: list = field(default_factory=list)
    aux_polys: list = field(default_factory=list)
    imag_roots: np.ndarray = None     # raízes do(s) polinômio(s) auxiliar(es) no eixo jw


def routh_table(coeffs, eps=1e-9, tol=1e-12):
    a = trim(coeffs)
    n = len(a) - 1
    if n < 1:
        raise ValueError("Polinômio de grau < 1.")
    cols = (n + 2) // 2
    R = np.zeros((n + 1, cols))
    R[0, :len(a[0::2])] = a[0::2]
    R[1, :len(a[1::2])] = a[1::2]
    notes, aux_list = [], []

    def fix_row(k):
        """Trata linha nula (aux. polinomial) ou zero isolado na 1ª coluna (epsilon)."""
        scale = max(1.0, float(np.max(np.abs(R[k - 1]))))
        if np.all(np.abs(R[k]) <= tol * scale):
            order = n - (k - 1)                       # potência de s da linha k-1
            aux = np.zeros(order + 1)
            for j in range(cols):
                p = order - 2 * j
                if p >= 0:
                    aux[order - p] = R[k - 1, j]
            aux_list.append(aux)
            for j in range(cols):
                p = order - 2 * j
                R[k, j] = R[k - 1, j] * p if p > 0 else 0.0
            notes.append(f"linha s^{n - k} nula: polinômio auxiliar de ordem {order}")
        elif abs(R[k, 0]) <= tol * scale:
            R[k, 0] = eps
            notes.append(f"zero na 1ª coluna da linha s^{n - k}: substituído por epsilon")

    fix_row(1)
    for k in range(2, n + 1):
        for j in range(cols - 1):
            R[k, j] = (R[k - 1, 0] * R[k - 2, j + 1] - R[k - 2, 0] * R[k - 1, j + 1]) / R[k - 1, 0]
        fix_row(k)
    return R, notes, aux_list


def routh_hurwitz(coeffs, eps=1e-9):
    R, notes, aux_list = routh_table(coeffs, eps)
    col = R[:, 0].copy()
    sgn = np.sign(col)
    sgn = sgn[sgn != 0]
    changes = int(np.sum(sgn[1:] != sgn[:-1]))
    imag = None
    if aux_list:
        r = np.concatenate([poly_roots(a) for a in aux_list])
        imag = r[np.abs(r.real) < 1e-6]
    if changes > 0:
        status = "instável"
    elif aux_list:
        status = "marginal"
    else:
        status = "estável"
    return RouthResult(R, col, changes, status, notes, aux_list, imag)


def routh_stability(G, K=1.0, pade_order=None):
    """Estabilidade de malha fechada de 1 + K G(s) = 0 via Routh.

    Para G com atraso é OBRIGATÓRIO informar pade_order (Routh não trata e^{-τs}).
    """
    if G.delay:
        if pade_order is None:
            raise ValueError("Routh não trata atraso: informe pade_order.")
        G = G.pade(pade_order)
    return routh_hurwitz(G.char_poly(K))


def print_table(res, n_dec=4):
    n = res.table.shape[0] - 1
    for i, row in enumerate(res.table):
        print(f"s^{n - i:<2d} | " + "  ".join(f"{v: .{n_dec}e}" if abs(v) > 1e4 or (v != 0 and abs(v) < 1e-3)
                                              else f"{v: .{n_dec}f}" for v in row))
    print("mudanças de sinal:", res.sign_changes, "->", res.status)
    for s in res.special_cases:
        print(" *", s)
