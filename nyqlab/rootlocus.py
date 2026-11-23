"""Lugar das Raízes (Root Locus) do zero: 1 + K G(s) = 0, K >= 0.

Os ramos são obtidos por CONTINUAÇÃO: as raízes para o ganho K_i servem de
chute inicial (Aberth) para K_{i+1}, o que mantém cada ramo contínuo.
"""
import numpy as np

from .polys import poly_roots, trim


def locus(G, Kmax=100.0, n=2000):
    """Retorna (Ks, R): R[i, j] = j-ésima raiz de D(s) + K_i N(s)."""
    G = G.rational()
    Ks = np.concatenate([[0.0], np.geomspace(Kmax * 1e-6, Kmax, n)])
    r = poly_roots(G.den)
    R = np.zeros((len(Ks), len(r)), dtype=complex)
    R[0] = r
    for i in range(1, len(Ks)):
        r = poly_roots(G.char_poly(Ks[i]), init=r)
        R[i] = r
    return Ks, R


def asymptotes(G):
    """Ângulos (graus) e centroide das assíntotas (n - m ramos que vão ao infinito)."""
    p, z = G.poles(), G.zeros()
    nm = len(p) - len(z)
    if nm <= 0:
        return np.array([]), 0.0
    centroid = (np.sum(p) - np.sum(z)).real / nm
    angles = np.array([(2 * q + 1) * 180.0 / nm for q in range(nm)])
    return angles, centroid


def breakaway_points(G):
    """Pontos de saída/entrada: raízes reais de N(s)D'(s) - N'(s)D(s) = 0 com K > 0."""
    G = G.rational()
    N, D = G.num, G.den
    poly = np.polysub(np.polymul(N, np.polyder(D)), np.polymul(np.polyder(N), D))
    out = []
    for s in poly_roots(poly):
        if abs(s.imag) < 1e-9:
            sr = s.real
            nval = np.polyval(N, sr)
            if abs(nval) > 1e-12:
                K = -np.polyval(D, sr) / nval
                if K > 0:
                    out.append((sr, K))
    return out


def critical_gain(G, Kmax=1e4, n=4000):
    """Menor K > 0 em que um ramo cruza o eixo jw (e a frequência do cruzamento)."""
    G = G.rational()

    def worst(K):
        r = poly_roots(G.char_poly(K))
        i = int(np.argmax(r.real))
        return r[i].real, abs(r[i].imag)

    Ks = np.concatenate([[0.0], np.geomspace(Kmax * 1e-6, Kmax, n)])
    prev = worst(Ks[0])[0]
    if prev >= 0:
        return None                                   # já começa sem estabilidade
    for a, b in zip(Ks[:-1], Ks[1:]):
        if worst(b)[0] >= 0:
            lo, hi = a, b
            for _ in range(100):
                m = 0.5 * (lo + hi)
                if worst(m)[0] >= 0:
                    hi = m
                else:
                    lo = m
            return 0.5 * (lo + hi), worst(0.5 * (lo + hi))[1]
    return None


def plot_locus(ax, G, Kmax=100.0, n=800, K_mark=None):
    Ks, R = locus(G, Kmax, n)
    ax.plot(R.real, R.imag, ".", ms=1.5)
    p, z = G.poles(), G.zeros()
    ax.plot(p.real, p.imag, "x", c="k", ms=8, mew=2)
    if z.size:
        ax.plot(z.real, z.imag, "o", mfc="none", c="k", ms=8, mew=1.5)
    if K_mark is not None:
        r = poly_roots(G.rational().char_poly(K_mark))
        ax.plot(r.real, r.imag, "s", c="r", ms=6)
    ax.axhline(0, c="gray", lw=0.5)
    ax.axvline(0, c="gray", lw=0.5)
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.set_title("Lugar das Raízes")
    ax.grid(alpha=0.3)
