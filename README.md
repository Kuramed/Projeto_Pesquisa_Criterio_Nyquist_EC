# nyqlab (versão Trilha EC) — Análise computacional de critérios de estabilidade (Nyquist, Routh-Hurwitz, Lugar das Raízes, Bode)

Projeto de Pesquisa Aplicada (CC/EC). Núcleo escrito **do zero com NumPy básico** (sem `control`, sem `scipy.signal`
no núcleo); SciPy / python-control só aparecem na validação cruzada.

## Instalação e execução
```
pip install -r requirements.txt
python tests/test_all.py            # 25 testes unitários (ou: python -m pytest -q)
python run_analysis.py              # 4 sistemas x 4 métodos + figuras em out/   (--show abre janelas)
python validate.py                  # Fase 3: validação cruzada (SciPy / python-control)
python benchmark.py --csv out/benchmark.csv   # Fase 3: escalabilidade n = 5, 10, 20
python ec_hil/hil_host.py --mode sim --sweep 0 0.1 0.3 0.36 0.45   # Trilha EC (sem hardware)
python ec_hil/hil_host.py --mode serial --port COM3 --sweep 0 0.1 0.3 0.36 0.45   # com a placa (carregue o .ino)
```

## Mapa: requisito do documento → arquivo
| Requisito | Onde |
|---|---|
| Princípio do Argumento / contorno de Nyquist (indentações em polos no eixo jω) | `nyqlab/nyquist.py::nyquist_contour` |
| Winding number (contagem de envoltórios em torno de −1+j0) | `encirclements_crossing` (geométrico) e `encirclements_angle` (verificação) |
| Z = N + P | `nyquist_analysis` |
| Margem de ganho / fase / atraso (automático) | `margins`, `delay_margin` |
| Routh-Hurwitz (zero na 1ª coluna e linha de zeros) + cálculo de polos | `nyqlab/routh.py`, `nyqlab/polys.py::poly_roots` (Aberth) |
| Lugar das Raízes (ramos, assíntotas, breakaway, K crítico) | `nyqlab/rootlocus.py` |
| Bode (exato e assintótico) | `nyqlab/bode.py` |
| Atraso puro exato vs Padé | `TF.pade`, `run_analysis.py::pade_vs_exato` |
| Sistemas A, B, C, D | `nyqlab/systems.py` |
| Validação cruzada | `validate.py` |
| Escalabilidade (tempo + erro numérico) | `benchmark.py` |
| Trilha EC: PID em ESP32/Arduino, HIL, atraso variável, limite crítico por Nyquist | `ec_hil/esp32_pid_hil/esp32_pid_hil.ino`, `ec_hil/hil_host.py` |
| Suíte de testes unitários | `tests/test_all.py` |

## Notas
* Convenção: `L(s) = K·G(s)`, realimentação unitária. `N` = envoltórios **horários**; `Z = N + P`.
* `margins()` devolve o cruzamento mais crítico; todos os cruzamentos vêm em `wgc_all`, `wpc_all`.
* Atraso crítico total (P = 0): `τ* = τ_atual + PM(rad)/ω_gc`. No sistema D: `2π/(3√3) ≈ 1,2092 s`.
* HIL: o laço discreto soma ≈ Ts/2 de atraso (ZOH); `hil_host.py` já subtrai isso da previsão.
* Firmware: protocolo texto `Y,k,ref,y` → `U,k,u`. O modo `sim` usa a mesma lei de controle (`PIDPython`).
