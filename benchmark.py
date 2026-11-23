"""Fase 3b: escalabilidade (tempo de CPU e erro numérico) para n = 5, 10, 20 (configurável).

Planta:  G_n(s) = Π p_i / Π (s + p_i)   (ganho DC = 1, todos os polos reais e estáveis).
Verdade conhecida por construção:
    K = 0.5  -> |L(jw)| <= 0.5 < 1  -> NUNCA envolve -1  -> malha fechada ESTÁVEL
    K = 1e3  -> n >= 3 -> malha fechada INSTÁVEL
Dois conjuntos de polos:
    'wilkinson' : p_i = 1..n         (mal condicionado: coeficientes até ~n!)
    'geometrico': p_i geométricos entre 0.5 e 8 (moderadamente condicionado)

Métricas por (conjunto, n):
  * tempo (mediana) de: Routh, raízes (Aberth próprio), np.roots, Nyquist (contagem N)
  * erro relativo máximo das raízes recuperadas a partir dos coeficientes arredondados
  * erro relativo de G(jw) avaliada pelos coeficientes (polyval) vs forma fatorada
  * se cada método acertou o veredito de estabilidade (K=0.5 e K=1e3)

Uso:  python benchmark.py [--orders 5 10 20] [--repeat 5] [--csv out/benchmark.csv]
"""
import argparse
import time

import numpy as np

from nyqlab import TF, poly_roots, routh_hurwitz, nyquist_analysis


def planta(kind, n):
    p = np.arange(1, n + 1, dtype=float) if kind == "wilkinson" else np.geomspace(0.5, 8, n)
    G = TF(np.array([np.prod(p)]), np.poly(-p))
    return p, G


def mediana(fn, rep):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def erro_raizes(est, true):
    est = list(est)
    err = 0.0
    for t in true:
        j = int(np.argmin([abs(e - t) for e in est]))
        err = max(err, abs(est[j] - t) / abs(t))
        est.pop(j)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    hdr = ("conjunto", "n", "t_routh[ms]", "t_aberth[ms]", "t_np.roots[ms]", "t_nyquist[ms]",
           "err_raizes_aberth", "err_raizes_np", "err_G(jw)_polyval", "routh_ok", "raizes_ok", "nyquist_ok")
    rows = []
    print(" | ".join(hdr))
    for kind in ("geometrico", "wilkinson"):
        for n in a.orders:
            p, G = planta(kind, n)
            w = np.geomspace(1e-2, 1e2, 400)
            exato = np.prod(p) / np.prod(1j * w[:, None] + p[None, :], axis=1)
            e_freq = float(np.max(np.abs(G.freq(w) - exato) / np.abs(exato)))
            e_ab = erro_raizes(poly_roots(G.den), -p)
            e_np = erro_raizes(np.roots(G.den), -p)
            t_routh = t_ab = t_np = t_ny = 0.0
            ok_r = ok_z = ok_n = 0
            for K, estavel in ((0.5, True), (1e3, False)):
                cp = G.char_poly(K)
                tr, rr = mediana(lambda: routh_hurwitz(cp), a.repeat)
                ta, ra = mediana(lambda: poly_roots(cp), a.repeat)
                tn, rn = mediana(lambda: np.roots(cp), a.repeat)
                ty, ny = mediana(lambda: nyquist_analysis(G, K), a.repeat)
                t_routh += tr; t_ab += ta; t_np += tn; t_ny += ty
                ok_r += (rr.status == "estável") == estavel
                ok_z += (not np.any(ra.real > 1e-9)) == estavel
                ok_n += (ny.Z == 0) == estavel
            row = (kind, n, 1e3 * t_routh / 2, 1e3 * t_ab / 2, 1e3 * t_np / 2, 1e3 * t_ny / 2,
                   e_ab, e_np, e_freq, f"{ok_r}/2", f"{ok_z}/2", f"{ok_n}/2")
            rows.append(row)
            print(" | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row))
    if a.csv:
        import csv, os
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(hdr); wr.writerows(rows)
        print("CSV salvo em", a.csv)


if __name__ == "__main__":
    main()
