"""Layerwise activation-input and minibatch weight-gradient measurements.

Snapshots use model copies; no optimizer step updates the trained model.
For no-BN models, dropout-off probes use the same trained checkpoint.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
ABS_THRESHOLDS = (0.25, 0.5)
PERCENTILES = (5, 25, 50, 75, 95)
_Z = torch.linspace(-10.0, 10.0, 40001, dtype=torch.float64)
_DZ = float(_Z[1] - _Z[0])
_W = torch.exp(-0.5 * _Z ** 2) / np.sqrt(2.0 * np.pi) * _DZ

def gaussian_A(activation, s):
    if not np.isfinite(s) or s <= 0:
        return float('nan')
    x = (s * _Z).clone().requires_grad_(True)
    with torch.enable_grad():
        grad, = torch.autograd.grad(activation(x).sum(), x)
    return float((grad * _W).sum())
PAPER_TABLE1 = {'Swish': (0.5, 1.0), 'Mish': (0.596996, 1.163508), 'BandHigh': (0.65126, 1.286958), 'BandLow': (0.557789, 1.104003)}

def check_against_table1(activations, tol=1e-05):
    print('Check of the A_f(s) integration against Table 1:')
    for name, (a_ref, r_ref) in PAPER_TABLE1.items():
        if name not in activations:
            continue
        act = activations[name]()
        a025 = gaussian_A(act, 0.25)
        ratio = a025 / gaussian_A(act, 3.0)
        print(f'  {name:9s} A(.25) = {a025:.6f} (paper {a_ref:.6f}) | R_A(.25,3) = {ratio:.6f} (paper {r_ref:.6f})')
        if abs(a025 - a_ref) > tol or abs(ratio - r_ref) > tol:
            raise RuntimeError(f'A_f(s) integration disagrees with Table 1 for {name}')

def make_fixed_batches(dataset, indices, batch_size, n_batches, device):
    batches = []
    for k in range(n_batches):
        idx = indices[k * batch_size:(k + 1) * batch_size]
        items = [dataset[int(i)] for i in idx]
        x = torch.stack([it[0] for it in items]).to(device)
        y = torch.tensor([int(it[1]) for it in items]).to(device)
        batches.append((x, y))
    return batches

def _col_name(prefix, value):
    return f"{prefix}{str(value).replace('.', 'p')}"

def layerwise_snapshot(model, batches, make_activation, diag_seed=0, use_dropout=True):
    has_norm = hasattr(model, 'norms')
    if not use_dropout and has_norm:
        raise ValueError('use_dropout=False would put BatchNorm in eval mode; it is only meant for models without BatchNorm')
    m = copy.deepcopy(model)
    m.train() if use_dropout else m.eval()
    pre_modules = list(m.norms) if has_norm else list(m.linears)
    n_layers = len(m.linears)
    device = next(m.parameters()).device
    acc = [dict(n=0, s1=0.0, s2=0.0, d1=0.0, vals=[]) for _ in range(n_layers)]

    def make_hook(i):

        def hook(module, inputs, output):
            x = output.detach().double()
            with torch.enable_grad():
                xr = x.clone().requires_grad_(True)
                fprime, = torch.autograd.grad(m.activation(xr).sum(), xr)
            a = acc[i]
            a['n'] += x.numel()
            a['s1'] += x.sum().item()
            a['s2'] += (x * x).sum().item()
            a['d1'] += fprime.sum().item()
            a['vals'].append(x.float().flatten().cpu())
        return hook
    handles = [mod.register_forward_hook(make_hook(i)) for i, mod in enumerate(pre_modules)]
    grad_norms = np.zeros(n_layers)
    fork_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(diag_seed)
        for x, y in batches:
            m.zero_grad(set_to_none=True)
            F.cross_entropy(m(x), y).backward()
            for i, lin in enumerate(m.linears):
                grad_norms[i] += lin.weight.grad.norm().item()
    grad_norms /= len(batches)
    for h in handles:
        h.remove()
    ref_act = make_activation()
    a_025 = gaussian_A(ref_act, 0.25)
    a_1 = gaussian_A(ref_act, 1.0)
    rows = []
    for i in range(n_layers):
        a = acc[i]
        mean = a['s1'] / a['n']
        var = max(a['s2'] / a['n'] - mean ** 2, 0.0)
        std = float(np.sqrt(var))
        if has_norm:
            gamma = m.norms[i].weight.detach().double()
            gamma_mean_abs = gamma.abs().mean().item()
            gamma_rms = gamma.pow(2).mean().sqrt().item()
        else:
            gamma_mean_abs = gamma_rms = float('nan')
        v = torch.cat(a['vals']).numpy()
        pct = np.percentile(v, PERCENTILES)
        abs_v = np.abs(v)
        row = {'layer': i + 1, 'preact_mean': mean, 'preact_std': std, 'preact_var': float(var)}
        for p, value in zip(PERCENTILES, pct):
            row[f'preact_q{p:02d}'] = float(value)
        for t in ABS_THRESHOLDS:
            row[_col_name('frac_abs_lt_', t)] = float((abs_v < t).mean())
        row.update({'gamma_mean_abs': gamma_mean_abs, 'gamma_rms': gamma_rms, 'emp_mean_fprime': a['d1'] / a['n'], 'A_at_preact_std': gaussian_A(ref_act, std), 'grad_norm_W': float(grad_norms[i]), 'A_at_0p25': a_025, 'A_at_1': a_1})
        rows.append(row)
        a['vals'] = None
    return rows
