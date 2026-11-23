"""Testes unitários. Rode:  python -m pytest -q   (ou)   python tests/test_all.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nyqlab import (TF, BENCHMARKS, poly_roots, pade, nyquist_analysis, margins, delay_margin,
                    encirclements_crossing, encirclements_angle, routh_hurwitz, routh_stability,
                    critical_gain, breakaway_points, asymptotes, locus, simulate_feedback,
                    asymptotic_magnitude)
from nyqlab.nyquist import nyquist_contour


def _same(a, b, tol=1e-8):
    """Compara multiconjuntos de raízes por casamento guloso (robusto a ordem)."""
    a = list(np.asarray(a, complex))
    b = list(np.asarray(b, complex))
    if len(a) != len(b):
        return False
    for x in a:
        j = int(np.argmin([abs(x - y) for y in b]))
        if abs(x - b[j]) > tol * (1 + abs(x)):
            return False
        b.pop(j)
    return True


# ---------------------------------------------------------------- polinômios
def test_roots_known():
    r = [-1, -2, -3, 4, -0.5 + 2j, -0.5 - 2j]
    p = np.poly(r)
    assert _same(poly_roots(p), r, 1e-8)


def test_roots_origin_multiple():
    p = np.polymul([1, 0, 0], [1, 3, 2])          # s^2 (s+1)(s+2)
    assert _same(poly_roots(p), [0, 0, -1, -2])


def test_roots_vs_numpy_random():
    rng = np.random.default_rng(0)
    for _ in range(30):
        p = rng.normal(size=rng.integers(3, 9))
        assert _same(poly_roots(p), np.roots(p), 1e-6)


def test_pade_first_order():
    n, d = pade(2.0, 1)
    assert np.allclose(n, [-1.0, 1.0]) and np.allclose(d, [1.0, 1.0])


# ---------------------------------------------------------------- Routh
def test_routh_basic():
    assert routh_hurwitz([1, 6, 11, 6]).status == "estável"
    assert routh_hurwitz([1, 6, 11, 100]).status == "instável"


def test_routh_matches_roots_random():
    rng = np.random.default_rng(1)
    bad = 0
    for _ in range(300):
        p = rng.normal(size=rng.integers(3, 9))
        rhp = int(np.sum(np.roots(p).real > 0))
        res = routh_hurwitz(p)
        bad += res.sign_changes != rhp
    assert bad == 0


def test_routh_zero_first_column_epsilon():
    p = [1, 2, 3, 6, 5, 3]                         # caso clássico (Ogata/Nise): 2 raízes no SPD
    res = routh_hurwitz(p)
    assert res.sign_changes == int(np.sum(np.roots(p).real > 0))
    assert any("epsilon" in s for s in res.special_cases)


def test_routh_zero_row_marginal():
    p = np.polymul(np.polymul([1, 0, 1], [1, 1]), [1, 2])   # (s²+1)(s+1)(s+2)
    res = routh_hurwitz(p)
    assert res.status == "marginal" and res.sign_changes == 0
    assert _same(np.sort(res.imag_roots.imag), [-1, 1])


def test_routh_zero_row_unstable():
    p = np.polymul([1, 0, 1], [1, -1, 1])           # (s²+1)(s²-s+1)  -> 2 raízes no SPD
    res = routh_hurwitz(p)
    assert res.sign_changes == 2 and res.status == "instável"


# ---------------------------------------------------------------- Nyquist
def test_benchmarks_Z():
    for k, b in BENCHMARKS.items():
        r = nyquist_analysis(b["G"], b["K"])
        assert r.Z == b["esperado_Z"] and r.N == r.N_angle, k


def test_nyquist_matches_roots_A_sweep():
    G = BENCHMARKS["A"]["G"]
    for K in [1, 10, 30, 59, 61, 80, 200]:
        z = int(np.sum(G.closed_loop_poles(K).real > 0))
        assert nyquist_analysis(G, K).Z == z, K


def test_nyquist_B_encirclement_direction():
    G = BENCHMARKS["B"]["G"]
    r = nyquist_analysis(G, 5.0)
    assert (r.P, r.N, r.Z) == (1, -1, 0)           # 1 volta anti-horária compensa P = 1
    assert nyquist_analysis(G, 1.0).Z == 1          # K < 2: instável


def test_nyquist_integrator_indentation():
    G = TF.from_zpk([], [0, -1, -2], 1.0)           # polo na origem
    for K, z in [(1.0, 0), (10.0, 2)]:               # Kcrit = 6
        assert nyquist_analysis(G, K).Z == z, K


def test_nyquist_imag_axis_pair():
    G = TF.from_zpk([], [1j, -1j, -2], 1.0)         # polos em ±j
    r = nyquist_analysis(G, 3.0)
    assert r.Z == int(np.sum(G.closed_loop_poles(3.0).real > 1e-9))


def test_margins_A_analytic():
    m = margins(BENCHMARKS["A"]["G"], 20.0)
    assert abs(m["GM"] - 3.0) < 1e-6 and abs(m["wpc"] - np.sqrt(11)) < 1e-6


def test_delay_critical_exact():
    G = BENCHMARKS["D"]["G"]
    dm, total = delay_margin(G, 2.0)
    assert abs(total - 2 * np.pi / (3 * np.sqrt(3))) < 1e-6
    assert nyquist_analysis(G.with_delay(total - 0.05), 2.0).Z == 0
    assert nyquist_analysis(G.with_delay(total + 0.05), 2.0).Z == 2


def test_pade_order1_misses_delay_limit():
    G = BENCHMARKS["D"]["G"].with_delay(1.5)         # exato: INSTÁVEL
    assert nyquist_analysis(G, 2.0).Z == 2
    assert routh_stability(G, 2.0, pade_order=1).status == "estável"    # Padé 1 erra
    assert routh_stability(G, 2.0, pade_order=6).status == "instável"   # Padé alto acerta


def test_crossing_vs_angle_random_curves():
    rng = np.random.default_rng(3)
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    for k in [1, 2, 3, -1, -2]:                      # k voltas anti-horárias em torno de -1
        L = -1 + (1 + 0.3 * rng.random()) * np.exp(1j * k * th)
        assert encirclements_crossing(L) == encirclements_angle(L) == -k


# ---------------------------------------------------------------- Root locus / Bode / tempo
def test_root_locus_features():
    G = BENCHMARKS["A"]["G"]
    K, w = critical_gain(G)
    assert abs(K - 60) < 1e-6 and abs(w - np.sqrt(11)) < 1e-6
    bp = breakaway_points(G)
    assert len(bp) == 1 and abs(bp[0][0] + 1.4226497) < 1e-5
    ang, c = asymptotes(G)
    assert _same(ang, [60, 180, 300]) and abs(c + 2) < 1e-12
    Kc = critical_gain(BENCHMARKS["C"]["G"])[0]
    assert abs(Kc - 60 / 7) < 1e-6


def test_locus_branches_continuous():
    Ks, R = locus(BENCHMARKS["A"]["G"], 100.0, 600)
    jumps = np.max(np.abs(np.diff(R, axis=0)))
    assert jumps < 1.0


def test_bode_asymptote_close():
    G = BENCHMARKS["A"]["G"]
    w = np.array([0.01, 100.0])
    assert np.allclose(asymptotic_magnitude(G, w), 20 * np.log10(np.abs(G.freq(w))), atol=0.2)


def test_step_final_value():
    G = BENCHMARKS["A"]["G"]
    t, y, _ = simulate_feedback(G, 20.0, T=30, dt=2e-3)
    assert abs(y[-1] - 20 / 26) < 1e-3
    t, y, _ = simulate_feedback(G, 100.0, T=30, dt=2e-3)
    assert abs(y[-1]) > 100 or len(t) < len(np.arange(0, 30, 2e-3))      # diverge


def test_step_delay_stability_matches_nyquist():
    G = BENCHMARKS["D"]["G"]
    _, y1, _ = simulate_feedback(G.with_delay(1.0), 2.0, T=80, dt=2e-3)
    _, y2, _ = simulate_feedback(G.with_delay(1.5), 2.0, T=80, dt=2e-3)
    assert abs(y1[-1] - 2 / 3) < 1e-2
    assert np.max(np.abs(y2[-4000:])) > 5 * np.max(np.abs(y1[-4000:]))


# ---------------------------------------------------------------- Trilha EC (HIL)
def _hil():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "ec_hil", "hil_host.py")
    spec = importlib.util.spec_from_file_location("hil_host", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hil_nyquist_prediction():
    h = _hil()
    pr = h.prever()
    for tau, esperado in ((pr["tau_crit_rede"] - 0.05, "estável"), (pr["tau_crit_rede"] + 0.05, "instável")):
        t, y, _ = h.run_loop(h.PIDPython(Ts=h.TS, **h.PID), lambda s, m=tau: m, T=60)
        assert h.classificar(t, y) == esperado, tau


def test_hil_serial_protocol_with_fake_board():
    """Emula a placa (mesma lei do firmware) atrás de uma porta serial falsa e compara com o modo sim."""
    h = _hil()

    class FakeSerial:
        def __init__(self, *a, **k):
            self.out, self.pid = [], None

        def reset_input_buffer(self): pass
        def close(self): pass

        def write(self, b):
            f = b.decode().strip().split(",")
            if f[0] == "CFG":
                Kp, Ki, Kd, Tf, Ts, um = map(float, f[1:])
                self.pid = h.PIDPython(Kp, Ki, Kd, Tf, Ts, um)
                self.out.append("OK\n")
            elif f[0] == "Y":
                u = self.pid.step(int(f[1]), float(f[2]), float(f[3]))
                self.out.append(f"U,{f[1]},{u:.6f}\n")

        def readline(self):
            return self.out.pop(0).encode()

    import types
    sys.modules["serial"] = types.SimpleNamespace(Serial=FakeSerial)
    import time
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        ref = h.PIDPython(Ts=h.TS, **h.PID)
        dut = h.PIDSerial("FAKE", 115200, Ts=h.TS, **h.PID)
        rng = np.random.default_rng(0)
        for k in range(200):
            y = float(f"{rng.normal():.6f}")           # a serial transmite 6 casas decimais
            assert abs(ref.step(k, 1.0, y) - dut.step(k, 1.0, y)) < 1e-5
    finally:
        time.sleep = real_sleep
        del sys.modules["serial"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:                 # noqa
                fails += 1
                print("FAIL", name, "->", repr(e))
    sys.exit(1 if fails else 0)
