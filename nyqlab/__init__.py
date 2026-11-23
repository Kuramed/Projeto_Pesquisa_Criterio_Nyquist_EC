"""nyqlab - análise de estabilidade (Nyquist, Routh-Hurwitz, Lugar das Raízes, Bode) do zero."""
from .tf import TF
from .polys import poly_roots, pade
from .nyquist import (nyquist_contour, nyquist_analysis, margins, delay_margin,
                      encirclements_crossing, encirclements_angle)
from .routh import routh_hurwitz, routh_stability, routh_table
from .rootlocus import locus, asymptotes, breakaway_points, critical_gain
from .bode import bode_data, asymptotic_magnitude
from .timesim import simulate_feedback, tf2ss
from .systems import BENCHMARKS
