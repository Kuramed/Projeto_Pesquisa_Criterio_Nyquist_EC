"""Critério de Estabilidade de Nyquist implementado do zero.

Passos:
 1. Contorno de Nyquist no plano s (eixo jw de -inf a +inf, com indentações
    semicirculares no semiplano direito em torno de polos sobre o eixo jw e
    grande semicírculo no infinito, que mapeia em um ponto para G própria).
 2. Mapeamento L(s) = K G(s) H(s) ao longo do contorno (Princípio do Argumento).
 3. Contagem de envoltórios horários N em torno de (-1, 0): método geométrico
    (cruzamentos do raio real x < -1) e método do ângulo acumulado (verificação).
 4. Z = N + P  (Z = polos instáveis de malha fechada).
O atraso e^{-tau s} é tratado EXATAMENTE (não há aproximação racional).
"""
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Contorno de Nyquist
# ----------------------------------------------------------------------------
def _axis_grid(lo, hi, lo_sing, hi_sing, n, eps, wmin):
    L = hi - lo
    if lo_sing and hi_sing:
        half = L / 2
        if half <= 2 * eps:
            return np.array([lo + eps, hi - eps])
        d = np.geomspace(eps, half, n // 2)
        return np.concatenate([lo + d, (hi - d)[::-1]])
    if lo_sing:
        return lo + np.geomspace(eps, L, n)
    if hi_sing:
        return (hi - np.geomspace(eps, L, n))[::-1]
    d = np.geomspace(wmin, hi, n)                 # sem polos no eixo: segmento contínuo
    return np.concatenate([-d[::-1], [0.0], d])


def _refine(fun, w, max_dphi=0.35, max_rel=0.2, max_iter=12):
    """Insere pontos onde o ângulo de (1+L) ou o módulo variam demais."""
    for _ in range(max_iter):
        L = fun(1j * w)
        z = 1.0 + L
        dphi = np.abs(np.angle(z[1:] * np.conj(z[:-1])))
        dL = np.abs(L[1:] - L[:-1]) / (1.0 + np.abs(L[1:]) + np.abs(L[:-1]))
        bad = np.flatnonzero((dphi > max_dphi) | (dL > max_rel))
        bad = bad[(w[bad + 1] - w[bad]) > 1e-12 * (1 + np.abs(w[bad]))]
        if bad.size == 0:
            break
        w = np.sort(np.concatenate([w, 0.5 * (w[bad] + w[bad + 1])]))
    return w


def nyquist_contour(G, K=1.0, wmin=1e-4, wmax=None, n=1500, eps=1e-4, refine=True, n_arc=200):
    """Retorna (s, L): pontos do contorno no plano s e imagem L = K*G(s)."""
    roots = np.concatenate([G.poles(), G.zeros()])
    scale = max(1.0, np.max(np.abs(roots))) if roots.size else 1.0
    if wmax is None:
        wmax = 1e3 * scale
    fun = lambda s: K * G(s)

    pw = np.sort(np.round(G.imag_axis_poles(), 9))
    pw = np.unique(pw)                                   # polos no eixo jw (multiplicidade vem de G)
    edges = [-wmax] + list(pw) + [wmax]
    s_parts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lo_sing, hi_sing = i > 0, i < len(edges) - 2
        w = _axis_grid(lo, hi, lo_sing, hi_sing, n, eps, wmin)
        if refine:
            w = _refine(fun, w)
        s_parts.append(1j * w)
        if hi_sing:                                       # indentação no semiplano direito
            th = np.linspace(-np.pi / 2, np.pi / 2, n_arc)
            s_parts.append(1j * hi + eps * np.exp(1j * th))
    s = np.concatenate(s_parts)
    return s, fun(s)


# ----------------------------------------------------------------------------
# Contagem de envoltórios
# ----------------------------------------------------------------------------
def encirclements_crossing(L, point=-1 + 0j):
    """N = nº de envoltórios HORÁRIOS (geométrico).

    Conta cruzamentos da semirreta que sai de `point` para a esquerda:
    cruzamento de baixo para cima = horário (+1); de cima para baixo = anti-horário (-1).
    """
    z = np.append(L, L[0]) - point
    x, y = z.real, z.imag
    up = (y[:-1] < 0) & (y[1:] >= 0)
    dn = (y[:-1] >= 0) & (y[1:] < 0)
    cross = up | dn
    dy = np.where(cross, y[1:] - y[:-1], 1.0)
    t = np.where(cross, -y[:-1] / dy, 0.0)
    xc = x[:-1] + t * (x[1:] - x[:-1])
    left = cross & (xc < 0)
    return int(np.sum(up & left) - np.sum(dn & left))


def encirclements_angle(L, point=-1 + 0j):
    """N pelo ângulo acumulado de (L - point) (verificação independente)."""
    z = np.append(L, L[0]) - point
    dphi = np.angle(z[1:] * np.conj(z[:-1]))
    return -int(round(dphi.sum() / (2 * np.pi)))


@dataclass
class NyquistResult:
    s: np.ndarray
    L: np.ndarray
    P: int
    N: int
    Z: int
    N_angle: int
    min_dist: float
    status: str                      # 'estável' | 'instável' | 'marginal'
    info: dict = field(default_factory=dict)


def nyquist_analysis(G, K=1.0, **kw):
    """Aplica o critério de Nyquist a L(s) = K G(s) (realimentação unitária)."""
    s, L = nyquist_contour(G, K, **kw)
    P = G.n_unstable_poles()
    N = encirclements_crossing(L)
    Na = encirclements_angle(L)
    dmin = float(np.min(np.abs(L + 1.0)))
    Z = N + P
    if dmin < 1e-3:
        status = "marginal"
    else:
        status = "estável" if Z == 0 else "instável"
    return NyquistResult(s, L, P, N, Z, Na, dmin, status)


# ----------------------------------------------------------------------------
# Margens de estabilidade
# ----------------------------------------------------------------------------
def _bisect(f, a, b, it=200):
    fa = f(a)
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
        if abs(b - a) < 1e-14 * (1 + abs(a)):
            break
    return 0.5 * (a + b)


def margins(G, K=1.0, wmin=1e-3, wmax=1e3, n=40000):
    """Margem de ganho/fase extraídas da resposta em frequência (sem toolbox).

    GM  : fator de ganho (e GM em dB) no cruzamento de fase mais crítico
          (aquele com |log GM| mínimo); wpc = frequência correspondente.
    PM  : margem de fase em graus no cruzamento de ganho mais crítico; wgc idem.
    Também são retornadas todas as frequências de cruzamento encontradas.
    """
    w = np.geomspace(wmin, wmax, n)
    if G.den[-1] != 0:                       # ω = 0 só é avaliável sem integrador em G
        w = np.concatenate([[0.0], w])
    L = K * G.freq(w)
    fun = lambda x: K * G.freq(x)

    # cruzamentos de ganho: |L| = 1
    f = np.abs(L) - 1.0
    idx = np.flatnonzero(f[:-1] * f[1:] < 0)
    wgc = np.array([_bisect(lambda x: abs(fun(x)) - 1.0, w[i], w[i + 1]) for i in idx])
    pms = []
    for wc in wgc:
        ph = np.degrees(np.angle(fun(wc)))
        pms.append(((ph + 180.0) + 180.0) % 360.0 - 180.0)

    # cruzamentos de fase: Im L = 0 com Re L < 0
    g = L.imag
    wpc = []
    if w[0] == 0.0 and g[0] == 0 and L[0].real < 0:
        wpc.append(0.0)
    idx = np.flatnonzero(g[:-1] * g[1:] < 0)
    for i in idx:
        wr = _bisect(lambda x: fun(x).imag, w[i], w[i + 1])
        if fun(wr).real < 0:
            wpc.append(wr)
    wpc = np.array(wpc)
    gms = np.array([1.0 / abs(fun(x)) for x in wpc]) if wpc.size else np.array([])

    out = dict(wgc_all=wgc, pm_all=np.array(pms), wpc_all=wpc, gm_all=gms,
               GM=np.inf, GM_dB=np.inf, wpc=np.nan, PM=np.inf, wgc=np.nan)
    if gms.size:
        k = np.argmin(np.abs(np.log(gms)))
        out.update(GM=gms[k], GM_dB=20 * np.log10(gms[k]), wpc=wpc[k])
    if len(pms):
        k = np.argmin(np.abs(pms))
        out.update(PM=pms[k], wgc=wgc[k])
    return out


def delay_margin(G, K=1.0, **kw):
    """Margem de atraso: atraso ADICIONAL tolerável antes da instabilidade.

    DM = min_k  PM_k(rad) / wgc_k  sobre os cruzamentos de ganho com PM_k > 0.
    O atraso crítico total é  G.delay + DM  (válido para P = 0).
    """
    m = margins(G, K, **kw)
    cand = [np.radians(pm) / wc for pm, wc in zip(m["pm_all"], m["wgc_all"]) if pm > 0 and wc > 0]
    dm = min(cand) if cand else np.inf
    return dm, G.delay + dm
