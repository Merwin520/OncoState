"""Patient-level split conformal calibration of the correspondence set. Nonconformity s = F(x_reference) - F*_0 (energy excess of the
reference matching over the V0 optimum). q = the ceil((n+1)(1-alpha))-th smallest calibration score (inf if that rank exceeds n).
Guarantee (exchangeability of patients): P(x_reference in {x : F(x) <= F*_0 + q}) >= 1 - alpha for a new patient."""
import math

import numpy as np


def quantile(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float('inf') if k > len(s) else float(s[k - 1])
