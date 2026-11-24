"""Trilha EC - Hardware-in-the-Loop com atraso de comunicação variável + previsão por Nyquist.

Arquitetura (o PC é a "planta + rede", a placa é o controlador):

   ref ──►(+)── e ──► [ PID no ESP32 ] ── u ──► atraso↓ ──► [ Planta simulada ] ──┐
         ▲                                                                       │
         └──────────────────────── atraso↑ ◄──────────────── y ◄────────────────┘

Modos:
  --mode sim     : PID em Python (mesma lei do firmware) -> roda sem hardware.
  --mode serial  : PID no ESP32/Arduino via porta serial (carregue esp32_pid_hil.ino).

Experimentos:
  --tau 0.2                  uma simulação com atraso de rede total (s) e perfil --profile
  --sweep 0 0.1 0.2 ...      varre atrasos e compara com a previsão do Nyquist
  --find-critical            bisseção do atraso crítico empírico (modo sim)

Previsão por Nyquist: L(s) = C(s) G(s) sem atraso -> margem de atraso DM = PM/wgc.
O laço discreto tem atraso extra ≈ Ts/2 (ZOH) que é subtraído da previsão (--extra).

Exemplos:
  python ec_hil/hil_host.py --mode sim --sweep 0 0.05 0.1 0.15 0.2 0.25 0.3
  python ec_hil/hil_host.py --mode sim --find-critical
  python ec_hil/hil_host.py --mode serial --port COM3 --tau 0.1 --profile sine --amp 0.05 --csv out/hil.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nyqlab import TF, delay_margin, margins, nyquist_analysis   
from nyqlab.timesim import tf2ss                                  

#  configuração
PLANTA = TF.from_zpk([], [-1, -2], 2.0)             # G(s) = 2/((s+1)(s+2)), ganho DC = 1
PID = dict(Kp=6.0, Ki=3.0, Kd=0.8, Tf=0.02)
TS = 0.01
UMAX = 100.0


def pid_tf(Kp, Ki, Kd, Tf):
    """C(s) = Kp + Ki/s + Kd s/(Tf s + 1)   (forma racional)."""
    return TF([Kp * Tf + Kd, Kp + Ki * Tf, Ki], [Tf, 1.0, 0.0])


#  controladores
class PIDPython:
    """Espelho exato do firmware (mesma discretização, anti-windup e saturação)."""

    def __init__(self, Kp, Ki, Kd, Tf, Ts, umax=UMAX):
        self.Kp, self.Ki, self.Kd, self.Tf, self.Ts, self.umax = Kp, Ki, Kd, Tf, Ts, umax
        self.reset()

    def reset(self):
        self.I = self.D = self.e_prev = 0.0

    def step(self, k, r, y):
        e = r - y
        self.I = float(np.clip(self.I + self.Ki * self.Ts * e, -self.umax, self.umax))
        self.D = self.Tf / (self.Tf + self.Ts) * self.D + self.Kd / (self.Tf + self.Ts) * (e - self.e_prev)
        self.e_prev = e
        return float(np.clip(self.Kp * e + self.I + self.D, -self.umax, self.umax))


class PIDSerial:
    """PID executado no ESP32/Arduino; o PC apenas troca mensagens de texto."""

    def __init__(self, port, baud, Kp, Ki, Kd, Tf, Ts, umax=UMAX, timeout=1.0):
        import serial                                  
        self.ser = serial.Serial(port, baud, timeout=timeout)
        import time
        time.sleep(2.0)                                 # reset automático da placa ao abrir a porta
        self.ser.reset_input_buffer()
        self.cfg = (Kp, Ki, Kd, Tf, Ts, umax)
        self.reset()

    def reset(self):
        Kp, Ki, Kd, Tf, Ts, umax = self.cfg
        self._cmd(f"CFG,{Kp},{Ki},{Kd},{Tf},{Ts},{umax}", expect="OK")

    def _cmd(self, text, expect):
        self.ser.write((text + "\n").encode())
        ans = self.ser.readline().decode().strip()
        if ans != expect:
            raise RuntimeError(f"Resposta inesperada da placa: {ans!r} (esperado {expect!r})")

    def step(self, k, r, y):
        self.ser.write(f"Y,{k},{r:.6f},{y:.6f}\n".encode())
        ans = self.ser.readline().decode().strip().split(",")
        if len(ans) != 3 or ans[0] != "U" or int(ans[1]) != k:
            raise RuntimeError(f"Mensagem inválida da placa: {ans}")
        return float(ans[2])

    def close(self):
        self.ser.close()


#  perfis de atraso
def delay_profile(kind, tau0, amp, T, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "const":
        return lambda t: tau0
    if kind == "ramp":                                    # cresce linearmente tau0 -> tau0+amp
        return lambda t: tau0 + amp * min(t / T, 1.0)
    if kind == "sine":
        return lambda t: max(0.0, tau0 + amp * np.sin(2 * np.pi * t / 10.0))
    if kind == "jitter":                                  # variação aleatória uniforme
        return lambda t: tau0 + amp * rng.random()
    raise ValueError(kind)


#  malha HIL
def rk4_hold(A, B, x, u, h):
    f = lambda z: A @ z + B[:, 0] * u
    k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run_loop(ctrl, tau_fn, T=60.0, Ts=TS, r=1.0, plant=PLANTA, sub=10):
    """Simula a malha: planta contínua (RK4, ZOH em u) + atraso de rede variável nos dois sentidos."""
    A, B, C, D = tf2ss(plant.num, plant.den)
    n = int(round(T / Ts))
    t = np.arange(n) * Ts
    y = np.zeros(n); u_calc = np.zeros(n); u_app = np.zeros(n)
    x = np.zeros(A.shape[0])
    ctrl.reset() if hasattr(ctrl, "reset") and not isinstance(ctrl, PIDSerial) else None
    if abs(D) > 0:
        raise ValueError("A planta HIL deve ser estritamente própria.")
    for k in range(n):
        y[k] = (C @ x).item()                           
        d = int(round(tau_fn(t[k]) / Ts))
        d_up, d_dn = d // 2, d - d // 2                 
        y_meas = y[k - d_up] if k >= d_up else 0.0      
        u_calc[k] = ctrl.step(k, r, y_meas)
        u_app[k] = u_calc[k - d_dn] if k >= d_dn else 0.0   
        for _ in range(sub):
            x = rk4_hold(A, B, x, u_app[k], Ts / sub)
        if abs(y[k]) > 1e6:                              
            return t[:k + 1], y[:k + 1], u_calc[:k + 1]
    return t, y, u_calc


def classificar(t, y, r=1.0):
    """'estável' se o erro decai; 'instável' se cresce/oscila sustentado ou diverge."""
    e = np.abs(r - y)
    if len(t) < 0.99 * (t[-1] + (t[1] - t[0])) / (t[1] - t[0]) or np.max(e) > 1e3:
        return "instável"
    T = t[-1]
    p1 = np.max(e[(t >= T - 20) & (t < T - 10)])
    p2 = np.max(e[t >= T - 10])
    return "instável" if (p2 > 1e-3 and p2 >= 0.98 * p1) else "estável"


#  previsão por Nyquist
def prever(Ts=TS, extra=None, planta=PLANTA, pid=PID):
    L = pid_tf(**pid) * planta                            # L(s) = C(s) G(s), SEM atraso
    m = margins(L)
    dm, _ = delay_margin(L)
    extra = Ts / 2 if extra is None else extra
    return dict(L=L, PM=m["PM"], wgc=m["wgc"], GM=m["GM"], DM=dm, extra=extra, tau_crit_rede=dm - extra)


def nyquist_verdict(tau_total, L):
    return nyquist_analysis(L.with_delay(tau_total), 1.0).status


#  experimentos
def make_ctrl(args):
    if args.mode == "serial":
        return PIDSerial(args.port, args.baud, Ts=args.Ts, **PID)
    return PIDPython(Ts=args.Ts, **PID)


def find_critical(args, lo=0.0, hi=1.5, iters=14):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ctrl = make_ctrl(args)
        t, y, _ = run_loop(ctrl, lambda s, m=mid: m, T=args.T, Ts=args.Ts)
        if classificar(t, y) == "estável":
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "serial"], default="sim")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--Ts", type=float, default=TS)
    ap.add_argument("--T", type=float, default=60.0, help="duração da simulação (s)")
    ap.add_argument("--tau", type=float, default=0.1, help="atraso de rede total base (s)")
    ap.add_argument("--profile", choices=["const", "ramp", "sine", "jitter"], default="const")
    ap.add_argument("--amp", type=float, default=0.05, help="amplitude da variação do atraso (s)")
    ap.add_argument("--sweep", type=float, nargs="+")
    ap.add_argument("--find-critical", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    pr = prever(args.Ts)
    print(f"Previsão (Nyquist) com L = C·G: PM = {pr['PM']:.2f}° em w_gc = {pr['wgc']:.3f} rad/s ; "
          f"GM = {pr['GM']:.2f}")
    print(f"  margem de atraso DM = {pr['DM']:.4f} s ; atraso extra do laço discreto ≈ {pr['extra']*1e3:.1f} ms")
    print(f"  => atraso CRÍTICO de rede previsto τ* = {pr['tau_crit_rede']:.4f} s\n")

    if args.sweep:
        print("τ_rede[s] | Nyquist (exato, c/ extra) | simulação HIL")
        for tau in args.sweep:
            ctrl = make_ctrl(args)
            t, y, _ = run_loop(ctrl, lambda s, m=tau: m, T=args.T, Ts=args.Ts)
            ny = nyquist_verdict(tau + pr["extra"], pr["L"])
            print(f"  {tau:7.3f} | {ny:9s}                 | {classificar(t, y)}")
        return
    if args.find_critical:
        tc = find_critical(args)
        print(f"Atraso crítico EMPÍRICO (bisseção na simulação) = {tc:.4f} s  |  previsto por Nyquist = "
              f"{pr['tau_crit_rede']:.4f} s  |  erro = {abs(tc - pr['tau_crit_rede']) * 1e3:.1f} ms")
        return

    ctrl = make_ctrl(args)
    t, y, u = run_loop(ctrl, delay_profile(args.profile, args.tau, args.amp, args.T),
                       T=args.T, Ts=args.Ts)
    print("Resultado:", classificar(t, y), f"(y_final = {y[-1]:.4f})")
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t", "y", "u"]); w.writerows(zip(t, y, u))
        print("CSV salvo em", args.csv)
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(t, y); ax[0].set_ylabel("y"); ax[0].grid(alpha=.3)
        ax[1].plot(t, u); ax[1].set_ylabel("u"); ax[1].set_xlabel("t (s)"); ax[1].grid(alpha=.3)
        plt.show()


if __name__ == "__main__":
    main()
