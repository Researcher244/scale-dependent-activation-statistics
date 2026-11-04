"""MNIST; BN; depth 35; additional five functions; five seeds; 100 epochs.
Checkpoint selection accepts any strict validation-loss improvement.

Outputs are isolated under runs/mnist_bn35_additional5.
Completed-run resume logic and scientific training settings are retained.
"""

import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
BASE_DIR = os.environ.get('ACTIVATION_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'mnist_bn35_additional5'))
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)
DATA_DIR = os.environ.get('ACTIVATION_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'))
RESULTS_PATH = os.path.join(BASE_DIR, 'C1RhoSlope__results.csv')
DIAG_PATH = os.path.join(BASE_DIR, 'C1RhoSlope__diagnostics.csv')
EPOCHS_PATH = os.path.join(BASE_DIR, 'C1RhoSlope__epochs.csv')
os.makedirs(DATA_DIR, exist_ok=True)
BATCH_SIZE = 128
BASE_LR = 0.01
ACT_LR_SCALE = 1.0
NUM_WORKERS = int(os.environ.get('ACTIVATION_NUM_WORKERS', '2'))
MIN_EPOCHS = 50
MAX_EPOCHS = 100
PATIENCE = 20
CORE_ACTIVATIONS = ['Logish', 'Smish', 'Gish', 'S-Mish', 'CoLU']
CORE_NORM_MODES = ['batchnorm']
CORE_DEPTHS = [35]
CORE_SEEDS = [42, 43, 44, 45, 46]

def get_patience(norm_mode):
    return PATIENCE

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
full_train_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
train_targets = np.array(full_train_dataset.targets)
split = StratifiedShuffleSplit(n_splits=1, test_size=1 / 6, random_state=42)
train_idx, val_idx = next(split.split(np.zeros(len(train_targets)), train_targets))
train_loader = DataLoader(Subset(full_train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(Subset(full_train_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f'Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_dataset)}')
_diag_rng = torch.Generator()
_diag_rng.manual_seed(0)
_diag_loader = DataLoader(Subset(full_train_dataset, train_idx), batch_size=256, shuffle=True, generator=_diag_rng, num_workers=0)
DIAG_INPUTS, DIAG_LABELS = next(iter(_diag_loader))
DIAG_INPUTS = DIAG_INPUTS.to(device)
DIAG_LABELS = DIAG_LABELS.to(device)
del _diag_loader, _diag_rng

class BatchNorm1dFixedGamma(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return self.bn(x) + self.shift

def make_norm_layer(mode, num_features):
    if mode == 'no_norm':
        return nn.Identity()
    elif mode == 'batchnorm':
        return nn.BatchNorm1d(num_features, affine=True)
    elif mode == 'batchnorm_fixed_gamma':
        return BatchNorm1dFixedGamma(num_features)
    elif mode == 'layernorm':
        return nn.LayerNorm(num_features, elementwise_affine=True)
    else:
        raise ValueError(f'Unknown norm mode: {mode}')

def get_norm_affine(norm_layer):
    if isinstance(norm_layer, BatchNorm1dFixedGamma):
        return (torch.ones_like(norm_layer.shift), norm_layer.shift.detach())
    if isinstance(norm_layer, nn.BatchNorm1d) and norm_layer.affine:
        return (norm_layer.weight.detach(), norm_layer.bias.detach())
    if isinstance(norm_layer, nn.LayerNorm) and norm_layer.elementwise_affine:
        return (norm_layer.weight.detach(), norm_layer.bias.detach())
    return (None, None)

def summarize_norm_scale(model):
    vals = []
    for norm in model.norms:
        gamma, _ = get_norm_affine(norm)
        if gamma is not None:
            vals.append(gamma.abs().mean().item())
    return round(float(np.mean(vals)), 4) if vals else float('nan')

def mish(x):
    return x * torch.tanh(F.softplus(x))

def swish(x):
    return x * torch.sigmoid(x)

class MishAct(nn.Module):

    def forward(self, x):
        return mish(x)

    def get_param(self):
        return {}

class SwishAct(nn.Module):

    def forward(self, x):
        return swish(x)

    def get_param(self):
        return {}

def gelu(x):
    return F.gelu(x)

class GeluAct(nn.Module):

    def forward(self, x):
        return gelu(x)

    def get_param(self):
        return {}

class FixC1MatchSlopeAndRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 1.0056290824
        self.alpha = 0.2481362485
        self.b = -0.6108569031

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixC1FixRhoMatchSlope(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 1.0655913532
        self.alpha = 0.2647998861
        self.b = -0.6847583639

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixC1FixSlopeMatchRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.9579116739
        self.alpha = 0.0
        self.b = 0.0

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

    def get_param(self):
        return {}

class FixRhoMatchC1AndSlope(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 2.5097657107
        self.alpha = -0.3622986508
        self.b = 3.239628968

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixRhoFixSlopeMatchC1(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 3.6454545499
        self.alpha = -0.4959679749
        self.b = 5.5094463356

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixSlopeMatchC1AndRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 3.2799942834
        self.alpha = -0.4929381821
        self.b = 4.9459659006

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixSlopeandC1HighRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.5
        self.alpha = 0.0
        self.b = 0.0

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixSlopeandC1LowRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 2.0
        self.alpha = 0.0
        self.b = 0.0

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class ScaledSwish(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = 1.4

    def forward(self, x):
        return self.scale * x * torch.sigmoid(x)

    def get_param(self):
        return {}

class HighBandSwish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = -1.0559
        self.b = -0.5061
        self.alpha = 0.1947
        self.gamma = -0.0708

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class LowBandSwish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = -1.0127
        self.b = 0.4625
        self.alpha = -0.1843
        self.gamma = 0.0707

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class BandHigh(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 1.20680028
        self.b = 0.04461946
        self.alpha = 0.16568067
        self.gamma = -0.07683369

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class BandLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 3.09266783
        self.b = 4.56834656
        self.alpha = -0.47355169
        self.gamma = 0.08382025

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class GentleTiltDown09(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 1.3212061309
        self.b = 0.6706517594
        self.alpha = -0.0277762821
        self.gamma = -0.1265

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class MatchAllToMish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 2.0996330505
        self.alpha = -0.3271993629
        self.b = 2.5444438979

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class ExpExpishAct(nn.Module):

    def forward(self, x):
        return x * torch.exp(-torch.exp(-x))

    def get_param(self):
        return {}

class SerfAct(nn.Module):

    def forward(self, x):
        return x * torch.erf(F.softplus(x))

    def get_param(self):
        return {}

class TanhExpAct(nn.Module):

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

    def get_param(self):
        return {}

class PhishAct(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.gelu(x))

    def get_param(self):
        return {}

class SquareSwishAct(nn.Module):

    def forward(self, x):
        s = torch.sigmoid(x)
        return x * s * s

    def get_param(self):
        return {}

class SetD1_LowA_HighGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.4949478828
        self.b = -0.413756283
        self.alpha = 0.0388663503
        self.eta = 0.0027885451

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD2_LowA_LowGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.42081386
        self.b = 0.48359119
        self.alpha = -0.18791026
        self.eta = -0.00805122

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD3_MidA_LowGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.3093398248
        self.b = 1.0044482442
        self.alpha = -0.223628617
        self.eta = -0.0137558189

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD4_MidA_HighGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.916394602
        self.b = 0.0436458135
        self.alpha = 0.0264917301
        self.eta = 0.0022294333

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD5_HighA_LowGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.25
        self.b = 1.6646126199
        self.alpha = -0.2443422477
        self.eta = -0.015827546

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD6_HighA_HighGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.3558473158
        self.b = 1.1158205221
        self.alpha = -0.1377719114
        self.eta = -0.009294866

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class ScaledSwishhalf(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = 0.5

    def forward(self, x):
        return self.scale * x * torch.sigmoid(x)

    def get_param(self):
        return {}

def _new_act_work(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

class LogishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        return (z * torch.log1p(torch.sigmoid(z))).to(x.dtype)

    def get_param(self):
        return {}

class SmishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        return (z * torch.tanh(torch.log1p(torch.sigmoid(z)))).to(x.dtype)

    def get_param(self):
        return {}

class GishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        inner = torch.exp(torch.clamp(z, max=20.0))
        return (z * torch.log1p(-torch.expm1(-inner))).to(x.dtype)

    def get_param(self):
        return {}

class SMishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        q = torch.sigmoid(z)
        return (z * (2.0 * q / (1.0 + q.square()))).to(x.dtype)

    def get_param(self):
        return {}

class CoLUAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        neg_x = torch.where(z < 0, z, torch.zeros_like(z))
        e = torch.exp(neg_x + torch.exp(neg_x))
        negative = neg_x * e / (e - neg_x)
        pos_x = torch.where(z >= 0, z, torch.zeros_like(z))
        t = pos_x * torch.exp(-pos_x - torch.exp(torch.clamp(pos_x, max=20.0)))
        positive = pos_x / (1.0 - t)
        return torch.where(z >= 0, positive, negative).to(x.dtype)

    def get_param(self):
        return {}
ACTIVATIONS = {'Logish': LogishAct, 'Smish': SmishAct, 'Gish': GishAct, 'S-Mish': SMishAct, 'CoLU': CoLUAct, 'Mish': MishAct, 'Swish': SwishAct, 'FixC1MatchSlopeAndRho': FixC1MatchSlopeAndRho, 'FixC1FixRhoMatchSlope': FixC1FixRhoMatchSlope, 'FixC1FixSlopeMatchRho': FixC1FixSlopeMatchRho, 'FixRhoMatchC1AndSlope': FixRhoMatchC1AndSlope, 'FixRhoFixSlopeMatchC1': FixRhoFixSlopeMatchC1, 'FixSlopeMatchC1AndRho': FixSlopeMatchC1AndRho, 'FixSlopeandC1HighRho': FixSlopeandC1HighRho, 'FixSlopeandC1LowRho': FixSlopeandC1LowRho, 'MatchAllToMish': MatchAllToMish, 'BandHigh': BandHigh, 'BandLow': BandLow, 'GentleTiltDown09': GentleTiltDown09, 'ScaledSwish': ScaledSwish, 'HighBandSwish': HighBandSwish, 'LowBandSwish': LowBandSwish, 'GeluAct': GeluAct, 'ExpExpishAct': ExpExpishAct, 'SerfAct': SerfAct, 'TanhExpAct': TanhExpAct, 'PhishAct': PhishAct, 'SquareSwishAct': SquareSwishAct, 'SetD5_HighA_LowGamma': SetD5_HighA_LowGamma, 'ScaledSwishhalf': ScaledSwishhalf, 'SetD1_LowA_HighGamma': SetD1_LowA_HighGamma, 'SetD2_LowA_LowGamma': SetD2_LowA_LowGamma, 'SetD3_MidA_LowGamma': SetD3_MidA_LowGamma, 'SetD4_MidA_HighGamma': SetD4_MidA_HighGamma, 'SetD6_HighA_HighGamma': SetD6_HighA_HighGamma}

class DeepMLP(nn.Module):

    def __init__(self, act_class, n_layers, norm_mode, hidden_dim=500):
        super().__init__()
        self.activation = act_class()
        self.n_layers = n_layers
        self.norm_mode = norm_mode
        in_dim = 784
        linears, norms = ([], [])
        for _ in range(n_layers):
            lin = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            linears.append(lin)
            norms.append(make_norm_layer(norm_mode, hidden_dim))
            in_dim = hidden_dim
        self.linears = nn.ModuleList(linears)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(hidden_dim, 10)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for lin, norm in zip(self.linears, self.norms):
            x = lin(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return self.output(x)

    def make_optimizer(self, base_lr, act_lr_scale):
        act_param_ids = {id(p) for p in self.activation.parameters()}
        base_params = [p for p in self.parameters() if id(p) not in act_param_ids]
        act_params = list(self.activation.parameters())
        groups = [{'params': base_params, 'lr': base_lr}]
        if act_params:
            groups.append({'params': act_params, 'lr': base_lr * act_lr_scale})
        return optim.SGD(groups)

def verify_shared_param(model, act_name):
    act_params = [n for n, p in model.named_parameters() if 'activation' in n]
    if act_params:
        assert len(act_params) == 1, f'Expected 1 shared param, found {len(act_params)}: {act_params}'
        print(f'  [OK] {act_name}: 1 shared param ({act_params[0]})')
    else:
        print(f'  [OK] {act_name}: no learnable activation params (fixed)')
print('\nVerifying shared parameter setup:')
for name in CORE_ACTIVATIONS:
    cls = ACTIVATIONS[name]
    m = DeepMLP(cls, n_layers=3, norm_mode='batchnorm')
    verify_shared_param(m, name)
    del m
print()

def pairwise_cosine_similarity_offdiag_mean(x):
    x_norm = F.normalize(x, dim=1, eps=1e-08)
    sim = x_norm @ x_norm.t()
    n = x.shape[0]
    off_diag_sum = sim.sum() - sim.diagonal().sum()
    return (off_diag_sum / (n * (n - 1))).item()

def collect_layer_stats(model, inputs, labels, criterion):
    model.eval()
    forward_rows = []
    h = inputs.view(inputs.size(0), -1)
    for i, (lin, norm) in enumerate(zip(model.linears, model.norms)):
        with torch.no_grad():
            h = lin(h)
            h = norm(h)
            z = h.clone()
        z_g = z.detach().requires_grad_(True)
        with torch.enable_grad():
            fz = model.activation(z_g)
            fz.sum().backward()
        df = z_g.grad.detach()
        act = fz.detach()
        pre_std = z.std().item()
        zc = z - z.mean()
        pre_skew = (zc ** 3).mean().item() / (pre_std ** 3 + 1e-08)
        pre_kurt = (zc ** 4).mean().item() / (pre_std ** 4 + 1e-08) - 3.0
        gamma, shift = get_norm_affine(norm)
        gamma_mean = gamma.mean().item() if gamma is not None else float('nan')
        gamma_absmean = gamma.abs().mean().item() if gamma is not None else float('nan')
        shift_mean = shift.mean().item() if shift is not None else float('nan')
        post_std = act.std().item()
        with torch.no_grad():
            cross_ex_corr_pre = pairwise_cosine_similarity_offdiag_mean(z)
            cross_ex_corr_post = pairwise_cosine_similarity_offdiag_mean(act)
        forward_rows.append({'layer': i + 1, 'pre_mean': z.mean().item(), 'pre_std': pre_std, 'pre_skew': pre_skew, 'pre_kurt': pre_kurt, 'post_mean': act.mean().item(), 'post_std': post_std, 'contraction_ratio': post_std / (pre_std + 1e-08), 'mean_deriv': df.mean().item(), 'deriv_sq_mean': (df ** 2).mean().item(), 'frac_lt1': (z.abs() < 1).float().mean().item(), 'gamma_mean': gamma_mean, 'gamma_absmean': gamma_absmean, 'shift_mean': shift_mean, 'cross_ex_corr_pre': cross_ex_corr_pre, 'cross_ex_corr_post': cross_ex_corr_post})
        with torch.no_grad():
            h = act
    model.zero_grad()
    with torch.enable_grad():
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
    grad_norms = []
    for lin in model.linears:
        gn = lin.weight.grad.norm().item() if lin.weight.grad is not None else float('nan')
        grad_norms.append(gn)
    model.zero_grad()
    result = []
    n = len(forward_rows)
    for i, row in enumerate(forward_rows):
        gn = grad_norms[i]
        gn_next = grad_norms[i + 1] if i + 1 < n else float('nan')
        ratio = gn / gn_next if not np.isnan(gn) and (not np.isnan(gn_next)) and (gn_next != 0) else float('nan')
        result.append({**row, 'grad_norm': gn, 'grad_norm_ratio': ratio})
    return result

def evaluate_acc(model, loader):
    model.eval()
    correct, total = (0, 0)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = (inputs.to(device), labels.to(device))
            _, predicted = torch.max(model(inputs), 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss, total = (0.0, 0)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = (inputs.to(device), labels.to(device))
            out = model(inputs)
            loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    return total_loss / total

def train_one_experiment(act_name, act_class, norm_mode, n_layers, seed):
    set_all_seeds(seed)
    model = DeepMLP(act_class, n_layers, norm_mode).to(device)
    optimizer = model.make_optimizer(BASE_LR, ACT_LR_SCALE)
    criterion = nn.CrossEntropyLoss()
    patience = get_patience(norm_mode)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    stopped_at = MAX_EPOCHS
    first_improvement_epoch = None
    baseline_val_loss = None
    best_path = os.path.join(BASE_DIR, f'_best_{act_name}_{norm_mode}_d{n_layers}_s{seed}.pt')
    epoch_rows = []
    print(f'\n>>> {act_name} | norm={norm_mode} | depth={n_layers} | seed={seed} | patience={patience}')
    start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = (inputs.to(device), labels.to(device))
            optimizer.zero_grad()
            criterion(model(inputs), labels).backward()
            optimizer.step()
        val_loss = evaluate_loss(model, val_loader, criterion)
        val_acc = evaluate_acc(model, val_loader)
        epoch_rows.append({'activation': act_name, 'norm_mode': norm_mode, 'n_layers': n_layers, 'seed': seed, 'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc})
        if epoch == 1:
            baseline_val_loss = val_loss
        if val_loss < best_val_loss:
            if first_improvement_epoch is None and epoch > 1 and (val_loss < baseline_val_loss):
                first_improvement_epoch = epoch
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
        if epoch == MIN_EPOCHS:
            epochs_no_improve = 0
        if epoch % 5 == 0 or epoch == 1:
            param_str = ' | '.join((f'{k}={v}' for k, v in model.activation.get_param().items()))
            gamma_str = f'mean|gamma|={summarize_norm_scale(model)}'
            print(f'   Epoch {epoch:02d}/{MAX_EPOCHS} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | patience {epochs_no_improve}/{patience} | {gamma_str}' + (f' | {param_str}' if param_str else ''))
        if epochs_no_improve >= patience and epoch >= MIN_EPOCHS:
            stopped_at = epoch
            print(f'   >>> Early stop at epoch {epoch}')
            break
    model.load_state_dict(torch.load(best_path))
    os.remove(best_path)
    test_acc = evaluate_acc(model, test_loader)
    converged = model.activation.get_param()
    gamma_final = summarize_norm_scale(model)
    val_acc_at_epoch_50 = next((r['val_acc'] for r in epoch_rows if r['epoch'] == 50), float('nan'))
    print(f'   FINAL test_acc={test_acc:.4f} | val_acc@50={val_acc_at_epoch_50:.4f} | best_val_loss={best_val_loss:.4f} | stopped_at={stopped_at} | mean|gamma|_final={gamma_final} | time={time.time() - start:.1f}s' + (f' | converged: {converged}' if converged else ''))
    criterion_diag = nn.CrossEntropyLoss()
    layer_stats = collect_layer_stats(model, DIAG_INPUTS, DIAG_LABELS, criterion_diag)
    for row in layer_stats:
        row.update({'activation': act_name, 'norm_mode': norm_mode, 'n_layers': n_layers, 'seed': seed})
    return (test_acc, converged, gamma_final, layer_stats, stopped_at, first_improvement_epoch, patience, epoch_rows, val_acc_at_epoch_50)
RUN_SPECS = []
for act in CORE_ACTIVATIONS:
    for norm_mode in CORE_NORM_MODES:
        for d in CORE_DEPTHS:
            for s in CORE_SEEDS:
                RUN_SPECS.append((act, norm_mode, d, s))
print(f'Total planned runs: {len(RUN_SPECS)}')
all_results = []
all_diag = []
all_epochs = []
if os.path.exists(RESULTS_PATH):
    try:
        all_results = pd.read_csv(RESULTS_PATH).to_dict('records')
        print(f'Loaded {len(all_results)} existing results from CSV.')
    except Exception as e:
        print(f'Could not load CSV ({e}). Starting fresh.')
if os.path.exists(DIAG_PATH):
    try:
        all_diag = pd.read_csv(DIAG_PATH).to_dict('records')
        print(f'Loaded {len(all_diag)} existing diagnostic rows from CSV.')
    except Exception as e:
        print(f'Could not load diagnostics CSV ({e}). Starting fresh.')
if os.path.exists(EPOCHS_PATH):
    try:
        all_epochs = pd.read_csv(EPOCHS_PATH).to_dict('records')
        print(f'Loaded {len(all_epochs)} existing epoch-trajectory rows from CSV.')
    except Exception as e:
        print(f'Could not load epochs CSV ({e}). Starting fresh.')
completed = {(str(r['activation']), str(r['norm_mode']), int(r['n_layers']), int(r['seed'])) for r in all_results}
total = len(RUN_SPECS)
done = len(completed)
print(f'Already done: {done}/{total}\n')
for act_name, norm_mode, n_layers, seed in RUN_SPECS:
    key = (act_name, norm_mode, n_layers, seed)
    if key in completed:
        print(f'Skipping {act_name} | {norm_mode} | depth={n_layers} | seed={seed}')
        continue
    act_class = ACTIVATIONS[act_name]
    test_acc, converged, gamma_final, layer_stats, stopped_at, first_improvement_epoch, patience_used, epoch_rows, val_acc_at_epoch_50 = train_one_experiment(act_name, act_class, norm_mode, n_layers, seed)
    done += 1
    row = {'activation': act_name, 'norm_mode': norm_mode, 'n_layers': n_layers, 'seed': seed, 'test_acc': round(test_acc, 4), 'val_acc_at_epoch_50': round(val_acc_at_epoch_50, 4) if np.isfinite(val_acc_at_epoch_50) else float('nan'), 'stopped_at': stopped_at, 'first_improvement_epoch': first_improvement_epoch, 'mean_abs_gamma_final': gamma_final, 'patience_used': patience_used}
    row.update({f'converged_{k}': v for k, v in converged.items()})
    all_results.append(row)
    pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)
    all_diag.extend(layer_stats)
    pd.DataFrame(all_diag).to_csv(DIAG_PATH, index=False)
    all_epochs.extend(epoch_rows)
    pd.DataFrame(all_epochs).to_csv(EPOCHS_PATH, index=False)
    print(f'[{done}/{total}] Saved.')
print('\n' + '=' * 60)
print('ALL DONE')
print('=' * 60)
df = pd.read_csv(RESULTS_PATH)
print('\nMean test accuracy by activation x norm_mode x depth:')
summary = df.groupby(['activation', 'norm_mode', 'n_layers'])['test_acc'].mean().round(4)
print(summary.to_string())
summary.reset_index().to_csv(os.path.join(BASE_DIR, 'C1RhoSlope_summary.csv'), index=False)
print('Saved C1RhoSlope_summary.csv')
if os.path.exists(DIAG_PATH):
    diag_df = pd.read_csv(DIAG_PATH)
    print('\nMean frac_lt1 and deriv_sq_mean at final layer, by activation x norm_mode x depth:')
    final_layer = diag_df[diag_df['layer'] == diag_df['n_layers']]
    fl_summary = final_layer.groupby(['activation', 'norm_mode', 'n_layers'])[['frac_lt1', 'pre_std', 'deriv_sq_mean', 'gamma_absmean']].mean().round(4)
    print(fl_summary.to_string())
    fl_summary.reset_index().to_csv(os.path.join(BASE_DIR, 'C1RhoSlope_final_layer_diagnostics_summary.csv'), index=False)
    print('Saved C1RhoSlope_final_layer_diagnostics_summary.csv')
