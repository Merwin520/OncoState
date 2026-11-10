"""Patient-level bootstrap summaries and paired comparisons."""
import numpy as np
from scipy.stats import wilcoxon


def patient_bootstrap(values, seed=20261001, n=2000, statistic=np.mean):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(estimate=None, ci_low=None, ci_high=None, n_patients=0)
    generator = np.random.default_rng(seed)
    bootstrap = np.array([statistic(generator.choice(values, len(values), replace=True)) for _ in range(n)])
    return dict(estimate=float(statistic(values)), ci_low=float(np.quantile(bootstrap, .025)),
                ci_high=float(np.quantile(bootstrap, .975)), n_patients=len(values))


def paired_bootstrap(first, second, seed=20261001, n=2000):
    first, second = np.asarray(first, float), np.asarray(second, float)
    keep = np.isfinite(first) & np.isfinite(second)
    difference = first[keep] - second[keep]
    result = patient_bootstrap(difference, seed, n)
    try:
        result['wilcoxon_p'] = float(wilcoxon(first[keep], second[keep]).pvalue) if len(difference) and np.any(difference) else 1.0
    except ValueError:
        result['wilcoxon_p'] = None
    return result
