"""Rebuild the manuscript's layerwise diagnostic figure from the supplied CSVs.

No training is run. Requires numpy, pandas, and matplotlib.
In Jupyter, run: %run diagnostics/make_diagnostic_figure.py
Paths are relative to this script, so the notebook's directory need not change.
"""
from pathlib import Path
import json
from math import erf, sqrt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / 'figures'
OUTPUT.mkdir(exist_ok=True)
FUNCTIONS = ['Mish', 'Swish', 'BandHigh', 'BandLow']
COLORS = dict(zip(FUNCTIONS, ['#0072B2', '#D55E00', '#009E73', '#8B4BA5']))
STYLES = dict(zip(FUNCTIONS, ['-', '--', '-.', ':']))
STAGES = ['init', 'epoch10', 'epoch20', 'epoch50', 'best']
bn = pd.read_csv(ROOT / 'data' / 'mnist_bn35_layerwise.csv')
nn = pd.read_csv(ROOT / 'data' / 'mnist_nobn15_layerwise.csv')
nn['dropout'] = nn['dropout'].astype(str).str.lower().map({'true': True, 'false': False})
assert nn['dropout'].notna().all()

def validate(df, layers, modes):
    assert set(df.activation) == set(FUNCTIONS)
    assert set(df.seed) == {42, 43}
    assert set(df.stage) == set(STAGES)
    keys = ['activation', 'seed', 'stage'] + (['dropout'] if modes == 2 else [])
    assert not df.duplicated(keys + ['layer']).any()
    assert len(df) == 4 * 2 * 5 * layers * modes
    for _, g in df.groupby(keys):
        assert set(g.layer) == set(range(1, layers + 1))
    cols = ['preact_std', 'preact_var', 'frac_abs_lt_0p25', 'grad_norm_W',
            'emp_mean_fprime', 'A_at_preact_std']
    assert np.isfinite(df[cols].to_numpy()).all()
    assert (df.preact_std > 0).all() and (df.grad_norm_W > 0).all()
    assert np.allclose(df.preact_std ** 2, df.preact_var)
    assert df.frac_abs_lt_0p25.between(0, 1).all()

validate(bn, 35, 1)
validate(nn, 15, 2)
on = nn.loc[nn.dropout].copy()

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
    'font.size': 8.4, 'axes.titlesize': 8.9, 'axes.labelsize': 8.3,
    'xtick.labelsize': 7.4, 'ytick.labelsize': 7.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'axes.linewidth': 0.65,
})

fig, axs = plt.subplots(3, 2, figsize=(6.30, 6.75))
fig.subplots_adjust(left=.115, right=.98, top=.895, bottom=.065,
                    hspace=.55, wspace=.40)
specs = [
    (axs[0, 0], bn, 'epoch10', 'preact_std', '(a) BN35: scale at epoch 10', 'Input standard deviation', False),
    (axs[0, 1], on, 'init', 'preact_std', '(b) No-BN15: initial scale', 'Input standard deviation', True),
    (axs[1, 0], bn, 'epoch10', 'frac_abs_lt_0p25', '(c) BN35: central mass at epoch 10', r'Fraction with $|h_\ell|<0.25$', False),
    (axs[1, 1], on, 'epoch10', 'preact_std', '(d) No-BN15: scale at epoch 10', 'Input standard deviation', True),
    (axs[2, 0], bn, 'init', 'grad_norm_W', '(e) BN35: initial weight gradients', r'Mean $\|\nabla_{W_\ell} L\|_F$', True),
    (axs[2, 1], on, 'best', 'preact_std', '(f) No-BN15: selected checkpoint', 'Input standard deviation', True),
]
for ax, df, stage, col, title, ylabel, log in specs:
    for function in FUNCTIONS:
        selected = df[(df.activation == function) & (df.stage == stage)]
        for seed, g in selected.groupby('seed'):
            g = g.sort_values('layer')
            ax.plot(g.layer, g[col], color=COLORS[function], lw=.7, alpha=.30,
                    linestyle=STYLES[function], zorder=1)
        means = selected.groupby('layer')[col].mean()
        ax.plot(means.index, means, color=COLORS[function], lw=1.65,
                linestyle=STYLES[function], zorder=3)
    ax.set(title=title, xlabel='Hidden layer', ylabel=ylabel)
    ax.set_xlim(1, int(df.layer.max()))
    ax.set_xticks([1, 10, 20, 30, 35] if df.layer.max() == 35 else [1, 5, 10, 15])
    if log:
        ax.set_yscale('log')
    else:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis='y', which='major', color='.9', lw=.6, zorder=0)
    ax.tick_params(length=3, width=.65)

axs[0, 0].axhline(1, color='.35', lw=.8, linestyle=':', zorder=0)
axs[0, 0].set_ylim(.987, 1.022)
gauss_mass = erf(.25 / sqrt(2))
axs[1, 0].axhline(gauss_mass, color='.35', lw=.8, linestyle=':', zorder=0)
axs[1, 0].text(34.5, gauss_mass-.025, r'$N(0,1)$ reference',
               fontsize=7, color='.30', ha='right', va='top')
axs[1, 0].set_ylim(.10, .77)
for ax in axs[:, 1]:
    ax.set_ylim(2e-4, 5)
# Validate fixed plotting limits against every seed, not just means.
for ax, df, stage, col, *_ in specs:
    values = df.loc[df.stage == stage, col]
    lo, hi = ax.get_ylim()
    assert values.min() >= lo and values.max() <= hi, (stage, col, lo, hi, values.min(), values.max())

handles = [Line2D([0], [0], color=COLORS[f], linestyle=STYLES[f], lw=1.8, label=f)
           for f in FUNCTIONS]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .995),
           ncol=4, frameon=False, handlelength=2.8, columnspacing=1.5)
fig.text(.53, .941, 'Thin lines: individual seeds   |   Thick lines: two-seed means',
         ha='center', fontsize=8, color='.3')
fig.savefig(OUTPUT / 'layerwise_diagnostics.pdf',
            metadata={'Title': 'Layerwise diagnostics for MNIST BN35 and no-BN15',
                      'Author': 'Chiheb Eddine Oualha and Mohammed Lakhdar Hadji'})
fig.savefig(OUTPUT / 'layerwise_diagnostics.png', dpi=200)
plt.close(fig)

# Compact, reproducible values cited in the manuscript, including the unused
# dropout-off probe mode. No endpoint performance is inferred from these files.
summaries = []
for label, df, depth in [('MNIST BN35', bn.assign(dropout=True), 35),
                         ('MNIST no-BN15', nn, 15)]:
    q = df[df.layer == depth].groupby(['activation', 'stage', 'dropout']).mean(numeric_only=True).reset_index()
    q.insert(0, 'setting', label)
    summaries.append(q)
pd.concat(summaries, ignore_index=True).to_csv(ROOT / 'last_layer_two_seed_means.csv', index=False)
bn[bn.layer == 1].groupby(['activation', 'stage']).grad_norm_W.mean().to_csv(
    ROOT / 'bn35_first_layer_gradient_means.csv')
audit = {
    'bn_rows': len(bn), 'no_bn_rows': len(nn), 'seeds': [42, 43],
    'functions': FUNCTIONS, 'stages': STAGES, 'independent_training_runs': 16,
    'bn_preact_std_range': [float(bn.preact_std.min()), float(bn.preact_std.max())],
    'bn_gamma_rms_range': [float(bn.gamma_rms.min()), float(bn.gamma_rms.max())],
    'standard_normal_mass_abs_lt_025': gauss_mass,
    'no_bn_dropout_modes': 'Two probes of the same trained checkpoints; not additional training runs.',
    'gradient_definition': 'Mean of four Frobenius norms, each from mean minibatch cross-entropy.',
}
(ROOT / 'measurement_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print('Created:', OUTPUT / 'layerwise_diagnostics.pdf')
