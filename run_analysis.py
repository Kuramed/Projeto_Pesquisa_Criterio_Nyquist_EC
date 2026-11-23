"""Fase 1-2: roda os 4 sistemas de benchmark com TODOS os métodos e gera as figuras em out/.

Uso:  python run_analysis.py            (salva PNGs em out/)
      python run_analysis.py --show     (também abre as janelas)
"""
import argparse
import os

import numpy as np

from nyqlab import (BENCHMARKS, nyquist_analysis, margins, delay_margin, routh_stability,
                    critical_gain, breakaway_points, asymptotes, simulate_feedback)
from nyqlab.rootlocus import plot_locus
from nyqlab.bode import plot_bode
from nyqlab.routh import print_table


def figuras(key, b, outdir, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G, K = b["G"], b["K"]
    r = nyquist_analysis(G, K)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax = axs[0, 0]
    ax.plot(r.L.real, r.L.imag, lw=1)
    ax.plot(-1, 0, "r+", ms=14, mew=2)
    lim = 3.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(f"Nyquist - {b['nome'][:40]}  (P={r.P}, N={r.N}, Z={r.Z})")
    ax.grid(alpha=0.3)
    plot_locus(axs[0, 1], G.rational(), Kmax=max(100, 5 * K), K_mark=K)
    axs[0, 1].set_xlim(-8, 4); axs[0, 1].set_ylim(-6, 6)
    bx = [plt.subplot(4, 2, 5), plt.subplot(4, 2, 7)]
    axs[1, 0].remove(); axs[1, 1].remove()
    plot_bode(bx, G, K)
    sx = fig.add_subplot(2, 2, 4)
    t, y, _ = simulate_feedback(G, K, T=30, dt=2e-3)
    sx.plot(t, np.clip(y, -10, 10)); sx.set_title("Resposta ao degrau (malha fechada)")
    sx.set_xlabel("t (s)"); sx.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, f"sistema_{key}.png")
    fig.savefig(path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
    return path


def relatorio(key, b):
    G, K = b["G"], b["K"]
    print("=" * 78)
    print(b["nome"], f"   [K = {K}]")
    print("=" * 78)
    r = nyquist_analysis(G, K)
    print(f"NYQUIST    : P={r.P}  N(horário)={r.N}  (verificação por ângulo: {r.N_angle})  "
          f"Z=N+P={r.Z}  -> {r.status}")
    m = margins(G, K)
    print(f"MARGENS    : GM={m['GM']:.4f} ({m['GM_dB']:.2f} dB) em w={m['wpc']:.4f}  |  "
          f"PM={m['PM']:.2f}° em w={m['wgc']:.4f}")
    if G.delay or key == "D":
        dm, tot = delay_margin(G, K)
        print(f"ATRASO     : margem de atraso adicional = {dm:.4f} s ; atraso crítico total = {tot:.4f} s")
    if G.delay:
        print("ROUTH      : NÃO trata e^{-τs} diretamente -> usa Padé:")
        for n in (1, 2, 4, 8):
            rr = routh_stability(G, K, pade_order=n)
            print(f"             Padé({n}) -> {rr.status} (mudanças de sinal = {rr.sign_changes})")
        print("RAÍZES     : (Lugar das Raízes só com Padé) crítico não é exato; ver pade_vs_exato abaixo")
    else:
        rr = routh_stability(G, K)
        print("ROUTH      : tabela")
        print_table(rr)
        print("POLOS MF   :", np.round(G.closed_loop_poles(K), 4))
        kc = critical_gain(G)
        bp = breakaway_points(G)
        ang, cen = asymptotes(G)
        print("LUGAR RAÍZES: K crítico =", None if kc is None else f"{kc[0]:.4f} (w={kc[1]:.4f})",
              "| assíntotas:", ang, f"centroide={cen:.3f}", "| breakaway:",
              [(round(float(s), 4), round(float(k), 4)) for s, k in bp])
    print()


def pade_vs_exato():
    """Demonstra a vantagem do Nyquist: atraso crítico exato vs. Padé de ordem n (Routh)."""
    print("=" * 78)
    print("SISTEMA D: atraso crítico — Nyquist (exato) vs Routh+Padé(n)")
    print("=" * 78)
    G = BENCHMARKS["D"]["G"]
    K = BENCHMARKS["D"]["K"]
    _, tot = delay_margin(G, K)
    print(f"Nyquist/margens (exato)      : τ_crit = {tot:.5f} s")
    for n in (1, 2, 3, 4, 6):
        lo, hi = 0.0, 10.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            st = routh_stability(G.with_delay(mid), K, pade_order=n).status
            if st == "estável":
                lo = mid
            else:
                hi = mid
        print(f"Routh + Padé({n})              : τ_crit = {0.5*(lo+hi):.5f} s   "
              f"(erro {abs(0.5*(lo+hi)-tot)/tot*100:.2f} %)")
    print("\nVeredito por atraso (K=2):  τ | Nyquist | Routh+Padé(1) | Routh+Padé(4)")
    for tau in (0.5, 1.0, 1.2, 1.3, 1.5, 1.9):
        ny = nyquist_analysis(G.with_delay(tau), K).status
        p1 = routh_stability(G.with_delay(tau), K, pade_order=1).status
        p4 = routh_stability(G.with_delay(tau), K, pade_order=4).status
        print(f"   {tau:4.1f} | {ny:9s} | {p1:9s} | {p4:9s}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)
    for k, b in BENCHMARKS.items():
        relatorio(k, b)
        print("figura:", figuras(k, b, "out", args.show), "\n")
    pade_vs_exato()
