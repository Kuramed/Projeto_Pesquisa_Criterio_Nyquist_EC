"""Fase 3a: validação cruzada do código próprio contra SciPy / python-control.

Compara: polos de malha fechada, margens (GM, PM, frequências), resposta em frequência
(incluindo atraso) e o veredito de estabilidade de Nyquist contra a verdade numérica
(raízes de alta precisão / Padé de ordem alta no sistema D).
Bibliotecas ausentes são ignoradas automaticamente (apenas informa).
"""
import numpy as np

from nyqlab import BENCHMARKS, nyquist_analysis, margins, poly_roots

try:
    import scipy.signal as ss
except ImportError:                                 
    ss = None
try:
    import control as ct
except ImportError:                                  
    ct = None


def _maxdiff(a, b):
    a = list(np.asarray(a, complex)); b = list(np.asarray(b, complex))
    return max(min(abs(x - y) for y in b) for x in a)


def linha(nome, ok, detalhe=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {nome} {detalhe}")
    return ok


def main():
    total, falhas = 0, 0

    def chk(nome, ok, det=""):
        nonlocal total, falhas
        total += 1
        falhas += 0 if linha(nome, ok, det) else 1

    for key, b in BENCHMARKS.items():
        G, K = b["G"], b["K"]
        print(f"\nSistema {key}  (K={K})")
        # 1) resposta em frequência (inclui o atraso exato)
        w = np.geomspace(1e-2, 1e2, 200)
        mine = K * G.freq(w)
        if ss is not None:
            _, h = ss.freqs(G.num * K, G.den, w)
            h = h * np.exp(-1j * w * G.delay)
            chk("resposta em frequência vs scipy.signal.freqs", np.allclose(mine, h, rtol=1e-9),
                f"(erro máx = {np.max(np.abs(mine - h)):.1e})")
        # 2) polos de malha fechada (racional; para D usa Padé 10 como referência)
        Gr = G.pade(10) if G.delay else G
        mp = poly_roots(Gr.char_poly(K))
        npoles = np.roots(Gr.char_poly(K))
        chk("polos de malha fechada vs np.roots", _maxdiff(mp, npoles) < 1e-6,
            f"(erro máx = {_maxdiff(mp, npoles):.1e})")
        # 3) veredito de Nyquist vs verdade numérica
        z_true = int(np.sum(npoles.real > 1e-9))
        z_nyq = nyquist_analysis(G, K).Z
        chk("Z de Nyquist == nº de polos instáveis de MF (verdade numérica)", z_nyq == z_true,
            f"(Nyquist Z={z_nyq}, referência={z_true})")
        # 4) margens vs python-control
        m = margins(G, K)
        if ct is not None and not G.delay:
            sysc = ct.tf(K * G.num, G.den)
            gm, pm, wcg, wcp = ct.margin(sysc)             # wcg: freq. de fase; wcp: freq. de ganho
            if np.isfinite(gm) and np.isfinite(m["GM"]):
                chk("GM vs python-control", abs(gm - m["GM"]) / gm < 1e-3,
                    f"(meu={m['GM']:.5f}, ct={gm:.5f})")
            if np.isfinite(pm) and np.isfinite(m["PM"]):
                chk("PM vs python-control", abs(pm - m["PM"]) < 0.05,
                    f"(meu={m['PM']:.3f}°, ct={pm:.3f}°)")
        elif ct is None:
            print("  (python-control não instalado: comparação de margens ignorada)")
        # 5) margens contra a solução analítica quando conhecida
        if key == "A":
            chk("GM analítico = 3", abs(m["GM"] - 3) < 1e-6)
        if key == "D":
            ang = np.degrees(np.pi - np.arctan(np.sqrt(3)) - np.sqrt(3) * G.delay)
            chk("PM analítico (sistema D)", abs(m["PM"] - ang) < 1e-6, f"(PM={m['PM']:.4f}°)")
    print(f"\nResumo: {total - falhas}/{total} verificações passaram.")
    return falhas


if __name__ == "__main__":
    raise SystemExit(main())
