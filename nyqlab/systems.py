"""Quatro plantas de benchmark (Fase 1) + ganho de malha K usado em cada cenário.

L(s) = K * G(s), realimentação unitária.
"""
from .tf import TF

BENCHMARKS = {
    "A": dict(
        nome="A - Padrão: estável em malha aberta, 3ª ordem",
        G=TF.from_zpk([], [-1, -2, -3], 1.0, name="A"), K=20.0,
        Kcrit=60.0, esperado_Z=0,
    ),
    "B": dict(
        nome="B - Instável em malha aberta (pêndulo invertido simplificado c/ atuador)",
        G=TF.from_zpk([], [1, -2], 1.0, name="B"), K=5.0,
        Kcrit=2.0, esperado_Z=0,           # estável para K > 2 (ganho mínimo)
    ),
    "C": dict(
        nome="C - Fase não-mínima: zero no semiplano direito (s = +1)",
        G=TF.from_zpk([1], [-1, -2, -3], -1.0, name="C"), K=7.0,
        Kcrit=60.0 / 7.0, esperado_Z=0,
    ),
    "D": dict(
        nome="D - Atraso de transporte: K e^{-0.5 s}/(s+1), K = 2",
        G=TF.from_zpk([], [-1], 1.0, delay=0.5, name="D"), K=2.0,
        Kcrit=None, esperado_Z=0,          # atraso crítico total = 2π/(3√3) ≈ 1.2092 s
    ),
}
