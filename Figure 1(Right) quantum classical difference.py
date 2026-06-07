import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -------------------------
# Parameters
# -------------------------
p_edge = 0.30
f_per_qubit = 0.985
N_G = 20
K = 100
M = 50
alpha = 0.05              # 95% confidence level for the quantum band
mc_samples = 200000       # Monte Carlo samples per n for the classical 95% line
mc_batch = 10000
mc_seed = 20260602
n_values = np.arange(8, 31, 1)


# -------------------------
# Curves
# -------------------------
def p_quantum(n: int, f: float) -> float:
    return 0.5 * (1.0 + f ** n)


def classical_instance_upper_bound_from_degrees(deg: np.ndarray) -> np.ndarray:
    """
    For one graph instance G, the analytic pointwise bound is
        p_cl^*(G) <= p_cl^UB(G) = 3/4 + 2^{-r(G)/2 - 2},
    where r(G) = max_i (d_i - 1_{d_i odd}).
    This function evaluates p_cl^UB(G) from the degree array.
    """
    r = np.max(deg - (deg % 2), axis=1)
    return 0.75 + np.power(2.0, -r / 2.0 - 2.0)


def estimate_classical_95_upper_line(n_values: np.ndarray,
                                     p: float,
                                     samples: int,
                                     batch: int,
                                     seed: int) -> np.ndarray:
    """
    Estimate the 95th percentile of the analytic per-instance upper bound
    p_cl^UB(G) over G ~ ER(n,p). Since p_cl^*(G) <= p_cl^UB(G) pointwise,
    this yields a 95% upper-confidence bound for p_cl^*.
    """
    q95 = []
    for n in n_values:
        rng = np.random.default_rng(seed + int(n))
        tri = np.triu_indices(n, 1)
        i, j = tri
        m = len(i)

        vals = np.empty(samples, dtype=np.float32)
        pos = 0
        for start in range(0, samples, batch):
            b = min(batch, samples - start)
            edges = (rng.random((b, m)) < p).astype(np.int8)
            A = np.zeros((b, n, n), dtype=np.int8)
            A[:, i, j] = edges
            A[:, j, i] = edges
            deg = A.sum(axis=2)
            vals[pos:pos + b] = classical_instance_upper_bound_from_degrees(deg)
            pos += b

        q95.append(float(np.quantile(vals, 0.95)))
    return np.array(q95)


# Quantum prediction and conservative 95% confidence band
pqu_curve = np.array([p_quantum(n, f_per_qubit) for n in n_values])
var_quantum_bound = pqu_curve * (1.0 - pqu_curve) / (N_G * K)
quantum_halfwidth = np.sqrt(var_quantum_bound / alpha)
q_lower = np.clip(pqu_curve - quantum_halfwidth, 0.0, 1.0)
q_upper = np.clip(pqu_curve + quantum_halfwidth, 0.0, 1.0)

# Classical 95% upper-confidence line
pcl95_curve = estimate_classical_95_upper_line(
    n_values=n_values,
    p=p_edge,
    samples=mc_samples,
    batch=mc_batch,
    seed=mc_seed,
)

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(7.1, 4.9))

# Quantum band first, behind the curves
ax.fill_between(
    n_values,
    q_lower,
    q_upper,
    color='tab:orange',
    alpha=0.16,
    linewidth=0,
    label='95% confidence interval for $p_{\\rm qu}$'
)

# Quantum line
ax.plot(
    n_values,
    pqu_curve,
    color='tab:orange',
    marker='s',
    markersize=4.4,
    linewidth=2.3,
    label=r'$p_{\rm qu}=(1+f^n)/2$, $f=0.985$'
)

# Classical high-probability upper line
ax.plot(
    n_values,
    pcl95_curve,
    color='tab:blue',
    linestyle='--',
    marker='o',
    markersize=4.1,
    linewidth=2.2,
    label=r'95% upper-confidence bound for $p_{\rm cl}^*$'
)

# Style
ax.set_xlabel('Qubit number $n$')
ax.set_ylabel('Winning probability')
ax.set_xlim(n_values.min() - 0.4, n_values.max() + 0.4)
ax.set_ylim(0.74, 0.97)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(np.arange(8, 31, 2))
ax.grid(alpha=0.22, linewidth=0.8)
ax.legend(frameon=False, loc='upper right', fontsize=9.0)

ax.text(
    0.02,
    0.03,
    rf'$N_G={N_G}$ graphs, $K={K}$ stabilizers/graph, $M={M}$ shots/stabilizer',
    transform=ax.transAxes,
    fontsize=9.0,
    va='bottom'
)

fig.tight_layout()
fig.savefig('/mnt/data/er_contextuality_classical95.pdf', bbox_inches='tight')
fig.savefig('/mnt/data/er_contextuality_classical95.png', dpi=220, bbox_inches='tight')
