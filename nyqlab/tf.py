"""Função de transferência LIT com atraso puro:  G(s) = N(s)/D(s) * e^{-tau s}."""
import numpy as np

from .polys import trim, poly_roots, pade, poly_from_roots


class TF:
    def __init__(self, num, den, delay=0.0, name=""):
        self.num = trim(num)
        self.den = trim(den)
        if len(self.num) > len(self.den):
            raise ValueError("Função de transferência imprópria (grau num > grau den).")
        self.delay = float(delay)
        self.name = name

    # ---------- construção ----------
    @classmethod
    def from_zpk(cls, zeros, poles, gain=1.0, delay=0.0, name=""):
        return cls(poly_from_roots(zeros, gain), poly_from_roots(poles), delay, name)

    # ---------- avaliação ----------
    def __call__(self, s):
        s = np.asarray(s, dtype=complex)
        g = np.polyval(self.num, s) / np.polyval(self.den, s)
        if self.delay:
            g = g * np.exp(-self.delay * s)
        return g

    def freq(self, w):
        """G(jw)."""
        return self(1j * np.asarray(w, dtype=float))

    # ---------- polos / zeros ----------
    def poles(self):
        return poly_roots(self.den)

    def zeros(self):
        return poly_roots(self.num)

    def n_unstable_poles(self, tol=1e-9):
        """P = número de polos de malha aberta no semiplano direito aberto."""
        return int(np.sum(self.poles().real > tol))

    def imag_axis_poles(self, tol=1e-7):
        p = self.poles()
        return p[np.abs(p.real) < tol * (1 + np.abs(p.imag))].imag

    # ---------- operações ----------
    def __mul__(self, other):
        if isinstance(other, TF):
            return TF(np.convolve(self.num, other.num), np.convolve(self.den, other.den),
                      self.delay + other.delay)
        return TF(self.num * float(other), self.den, self.delay, self.name)

    __rmul__ = __mul__

    def rational(self):
        return TF(self.num, self.den, 0.0, self.name)

    def with_delay(self, tau):
        return TF(self.num, self.den, tau, self.name)

    def pade(self, n):
        """Substitui e^{-tau s} pela aproximação de Padé de ordem n (parte racional)."""
        if not self.delay:
            return self.rational()
        pn, pd = pade(self.delay, n)
        return TF(np.convolve(self.num, pn), np.convolve(self.den, pd), 0.0, self.name)

    def char_poly(self, K=1.0):
        """Polinômio característico de malha fechada D(s) + K N(s) (só racional)."""
        if self.delay:
            raise ValueError("Use .pade(n) antes: o polinômio não representa o atraso.")
        n = max(len(self.den), len(self.num))
        d = np.concatenate([np.zeros(n - len(self.den)), self.den])
        m = np.concatenate([np.zeros(n - len(self.num)), self.num])
        return trim(d + K * m)

    def closed_loop_poles(self, K=1.0):
        return poly_roots(self.char_poly(K))

    def __repr__(self):
        return f"TF(num={self.num.tolist()}, den={self.den.tolist()}, delay={self.delay})"
