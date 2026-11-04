"""CIFAR-10 plain CNN; BN depths 30 and 40; five additional functions.

Five seeds per activation; source training and resume rules retained.
"""

import os
import sys
import time
import random
import traceback
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
NEW_FUNCTIONS_ONLY = True
NEW_ACTIVATIONS = ['Logish', 'Smish', 'Gish', 'S-Mish', 'CoLU']
WORK_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
BASE_DIR = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'runs/cifar_bn30_bn40_additional5')
__import__('pathlib').Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
DATA_DIR = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'datasets')
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
BATCH_SIZE = 256
BASE_LR = 0.01
ACT_LR_SCALE = 1.0
NUM_WORKERS = 2
MIN_EPOCHS = 30
MAX_EPOCHS = 60
PATIENCE = 15
ORIGINAL_ACTIVATIONS = ['Mish', 'Swish', 'FixC1MatchSlopeAndRho', 'FixC1FixRhoMatchSlope', 'FixC1FixSlopeMatchRho', 'FixRhoMatchC1AndSlope', 'FixRhoFixSlopeMatchC1', 'FixSlopeMatchC1AndRho', 'FixSlopeandC1HighRho', 'FixSlopeandC1LowRho', 'MatchAllToMish', 'BandHigh', 'BandLow', 'GentleTiltDown09', 'ScaledSwish', 'HighBandSwish', 'LowBandSwish', 'GeluAct', 'TanhExpAct', 'ExpExpishAct', 'SerfAct', 'PhishAct', 'SquareSwishAct', 'SetD5_HighA_LowGamma', 'ScaledSwishhalf', 'SetD1_LowA_HighGamma', 'SetD2_LowA_LowGamma', 'SetD3_MidA_LowGamma', 'SetD4_MidA_HighGamma', 'SetD6_HighA_HighGamma']
CORE_ACTIVATIONS = NEW_ACTIVATIONS if NEW_FUNCTIONS_ONLY else ORIGINAL_ACTIVATIONS + NEW_ACTIVATIONS
CORE_SEEDS = [42, 43, 44, 45, 46]
CORE_DEPTHS = [30, 40]
NORM_MODE = 'batchnorm'
print(f'PLAN: {len(CORE_ACTIVATIONS)} activations x {len(CORE_SEEDS)} seeds x {len(CORE_DEPTHS)} setting(s). Output: {BASE_DIR}', flush=True)
DEAD_UNIT_THRESHOLD = 0.001
WEIGHT_NORM_EPOCHS = {1, 10, 25, 50}

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
N_GPUS = torch.cuda.device_count()
print(f'Visible GPUs: {N_GPUS}')
if N_GPUS < 2:
    print('WARNING: fewer than 2 GPUs visible -- both depths will run on the same device sequentially (still correct, just no parallel speedup).')
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.247, 0.2435, 0.2616)
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
full_train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
full_train_dataset_eval = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=eval_transform)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=eval_transform)
train_targets = np.array(full_train_dataset.targets)
split = StratifiedShuffleSplit(n_splits=1, test_size=1 / 6, random_state=42)
train_idx, val_idx = next(split.split(np.zeros(len(train_targets)), train_targets))

def make_loaders():
    mp_ctx = 'spawn'
    train_loader = DataLoader(Subset(full_train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, multiprocessing_context=mp_ctx, persistent_workers=True)
    val_loader = DataLoader(Subset(full_train_dataset_eval, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, multiprocessing_context=mp_ctx, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, multiprocessing_context=mp_ctx, persistent_workers=True)
    return (train_loader, val_loader, test_loader)
print(f'Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_dataset)}')
_diag_rng = torch.Generator()
_diag_rng.manual_seed(0)
_diag_loader = DataLoader(Subset(full_train_dataset_eval, train_idx), batch_size=256, shuffle=True, generator=_diag_rng, num_workers=0)
DIAG_INPUTS_CPU, DIAG_LABELS_CPU = next(iter(_diag_loader))
del _diag_loader, _diag_rng

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

class GeluAct(nn.Module):

    def forward(self, x):
        return F.gelu(x)

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
        return (z * torch.log1p(torch.sigmoid(z))).to(dtype=x.dtype)

    def get_param(self):
        return {}

class SmishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        return (z * torch.tanh(torch.log1p(torch.sigmoid(z)))).to(dtype=x.dtype)

    def get_param(self):
        return {}

class GishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        inner = torch.exp(torch.clamp(z, max=20.0))
        gate = torch.log1p(-torch.expm1(-inner))
        return (z * gate).to(dtype=x.dtype)

    def get_param(self):
        return {}

class SMishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        q = torch.sigmoid(z)
        return (z * (2.0 * q / (1.0 + q.square()))).to(dtype=x.dtype)

    def get_param(self):
        return {}

class CoLUAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        neg_x = torch.where(z < 0.0, z, torch.zeros_like(z))
        e = torch.exp(neg_x + torch.exp(neg_x))
        negative = neg_x * e / (e - neg_x)
        pos_x = torch.where(z >= 0.0, z, torch.zeros_like(z))
        t = pos_x * torch.exp(-pos_x - torch.exp(torch.clamp(pos_x, max=20.0)))
        positive = pos_x / (1.0 - t)
        return torch.where(z >= 0.0, positive, negative).to(dtype=x.dtype)

    def get_param(self):
        return {}
ACTIVATIONS = {'Mish': MishAct, 'Swish': SwishAct, 'FixC1MatchSlopeAndRho': FixC1MatchSlopeAndRho, 'FixC1FixRhoMatchSlope': FixC1FixRhoMatchSlope, 'FixC1FixSlopeMatchRho': FixC1FixSlopeMatchRho, 'FixRhoMatchC1AndSlope': FixRhoMatchC1AndSlope, 'FixRhoFixSlopeMatchC1': FixRhoFixSlopeMatchC1, 'FixSlopeMatchC1AndRho': FixSlopeMatchC1AndRho, 'FixSlopeandC1HighRho': FixSlopeandC1HighRho, 'FixSlopeandC1LowRho': FixSlopeandC1LowRho, 'MatchAllToMish': MatchAllToMish, 'BandHigh': BandHigh, 'BandLow': BandLow, 'GentleTiltDown09': GentleTiltDown09, 'ScaledSwish': ScaledSwish, 'HighBandSwish': HighBandSwish, 'LowBandSwish': LowBandSwish, 'GeluAct': GeluAct, 'ExpExpishAct': ExpExpishAct, 'SerfAct': SerfAct, 'TanhExpAct': TanhExpAct, 'PhishAct': PhishAct, 'SquareSwishAct': SquareSwishAct, 'SetD5_HighA_LowGamma': SetD5_HighA_LowGamma, 'ScaledSwishhalf': ScaledSwishhalf, 'SetD1_LowA_HighGamma': SetD1_LowA_HighGamma, 'SetD2_LowA_LowGamma': SetD2_LowA_LowGamma, 'SetD3_MidA_LowGamma': SetD3_MidA_LowGamma, 'SetD4_MidA_HighGamma': SetD4_MidA_HighGamma, 'SetD6_HighA_HighGamma': SetD6_HighA_HighGamma}
ACTIVATIONS.update({'Logish': LogishAct, 'Smish': SmishAct, 'Gish': GishAct, 'S-Mish': SMishAct, 'CoLU': CoLUAct})
STAGE_CHANNELS = [32, 64, 128, 256, 256]

def stage_layout(n_layers, n_stages=5):
    base = n_layers // n_stages
    rem = n_layers % n_stages
    return [base + (1 if i < rem else 0) for i in range(n_stages)]

class DeepCNN(nn.Module):

    def __init__(self, act_class, n_layers, norm_mode='batchnorm', n_stages=5):
        super().__init__()
        assert n_layers >= n_stages, 'need at least 1 conv layer per stage'
        self.activation = act_class()
        self.n_layers = n_layers
        self.norm_mode = norm_mode
        layout = stage_layout(n_layers, n_stages)
        convs, norms, downsamples = ([], [], [])
        in_c = 3
        for stage_i, n_convs_in_stage in enumerate(layout):
            out_c = STAGE_CHANNELS[stage_i]
            for j in range(n_convs_in_stage):
                conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
                nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
                convs.append(conv)
                norms.append(nn.BatchNorm2d(out_c, affine=True))
                in_c = out_c
            if stage_i < n_stages - 1:
                downsamples.append(nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1, bias=False))
            else:
                downsamples.append(None)
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.downsamples = nn.ModuleList([d for d in downsamples if d is not None])
        self._layout = layout
        self._n_stages = n_stages
        self.dropout = nn.Dropout2d(0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(in_c, 10)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        idx = 0
        ds_idx = 0
        for stage_i, n_convs_in_stage in enumerate(self._layout):
            for _ in range(n_convs_in_stage):
                x = self.convs[idx](x)
                x = self.norms[idx](x)
                x = self.activation(x)
                x = self.dropout(x)
                idx += 1
            if stage_i < self._n_stages - 1:
                x = self.downsamples[ds_idx](x)
                ds_idx += 1
        x = self.pool(x).flatten(1)
        return self.output(x)

    def make_optimizer(self, base_lr, act_lr_scale):
        act_param_ids = {id(p) for p in self.activation.parameters()}
        base_params = [p for p in self.parameters() if id(p) not in act_param_ids]
        act_params = list(self.activation.parameters())
        groups = [{'params': base_params, 'lr': base_lr}]
        if act_params:
            groups.append({'params': act_params, 'lr': base_lr * act_lr_scale})
        return optim.SGD(groups, momentum=0.9, weight_decay=0.0005, nesterov=True)

def pairwise_cosine_similarity_offdiag_mean(x):
    x_flat = x.flatten(1)
    x_norm = F.normalize(x_flat, dim=1, eps=1e-08)
    sim = x_norm @ x_norm.t()
    n = x_flat.shape[0]
    off_diag_sum = sim.sum() - sim.diagonal().sum()
    return (off_diag_sum / (n * (n - 1))).item()

def spectral_and_frobenius(weight):
    w2d = weight.detach().reshape(weight.shape[0], -1)
    fro = w2d.norm().item()
    try:
        sv1 = torch.linalg.svdvals(w2d)[0].item()
    except Exception:
        sv1 = float('nan')
    return (fro, sv1)

def collect_layer_stats(model, inputs, labels, criterion, device, bn_mode='eval'):
    if bn_mode == 'eval':
        model.eval()
    else:
        model.train()
        model.dropout.eval()
    rows = []
    x = inputs
    idx = 0
    ds_idx = 0
    for stage_i, n_convs_in_stage in enumerate(model._layout):
        for _ in range(n_convs_in_stage):
            with torch.no_grad():
                x = model.convs[idx](x)
                x = model.norms[idx](x)
                z = x.clone()
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
            gamma = model.norms[idx].weight.detach()
            shift = model.norms[idx].bias.detach()
            frac_dead = (df.abs() < DEAD_UNIT_THRESHOLD).float().mean().item()
            with torch.no_grad():
                cross_ex_corr_pre = pairwise_cosine_similarity_offdiag_mean(z)
                cross_ex_corr_post = pairwise_cosine_similarity_offdiag_mean(act)
            rows.append({'layer': idx + 1, 'bn_mode': bn_mode, 'pre_mean': z.mean().item(), 'pre_std': pre_std, 'pre_skew': pre_skew, 'pre_kurt': pre_kurt, 'post_mean': act.mean().item(), 'post_std': act.std().item(), 'contraction_ratio': act.std().item() / (pre_std + 1e-08), 'mean_deriv': df.mean().item(), 'deriv_sq_mean': (df ** 2).mean().item(), 'frac_dead': frac_dead, 'frac_lt1': (z.abs() < 1).float().mean().item(), 'gamma_mean': gamma.mean().item(), 'gamma_absmean': gamma.abs().mean().item(), 'shift_mean': shift.mean().item(), 'cross_ex_corr_pre': cross_ex_corr_pre, 'cross_ex_corr_post': cross_ex_corr_post})
            with torch.no_grad():
                x = act
            idx += 1
        if stage_i < model._n_stages - 1:
            with torch.no_grad():
                x = model.downsamples[ds_idx](x)
            ds_idx += 1
    with torch.no_grad():
        logits = model.pool(x).flatten(1)
        logits = model.output(logits)
        logit_std = logits.std().item()
        logit_cross_corr = pairwise_cosine_similarity_offdiag_mean(logits)
    model.zero_grad()
    with torch.enable_grad():
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
    grad_norms = [c.weight.grad.norm().item() if c.weight.grad is not None else float('nan') for c in model.convs]
    model.zero_grad()
    n = len(rows)
    result = []
    for i, row in enumerate(rows):
        gn = grad_norms[i]
        gn_next = grad_norms[i + 1] if i + 1 < n else float('nan')
        ratio = gn / gn_next if not np.isnan(gn) and (not np.isnan(gn_next)) and (gn_next != 0) else float('nan')
        result.append({**row, 'grad_norm': gn, 'grad_norm_ratio': ratio, 'logit_std': logit_std, 'logit_cross_ex_corr': logit_cross_corr})
    return result

def collect_weight_norms(model, epoch):
    rows = []
    for i, conv in enumerate(model.convs):
        fro, sv1 = spectral_and_frobenius(conv.weight)
        rows.append({'epoch': epoch, 'layer': i + 1, 'frobenius_norm': fro, 'spectral_norm': sv1})
    return rows

def evaluate_acc(model, loader, device):
    model.eval()
    correct, total = (0, 0)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = (inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(inputs)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total = (0.0, 0)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = (inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(inputs)
                loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    return total_loss / total

def train_one_experiment(act_name, act_class, norm_mode, n_layers, seed, device):
    set_all_seeds(seed)
    train_loader, val_loader, test_loader = make_loaders()
    model = DeepCNN(act_class, n_layers, norm_mode).to(device)
    optimizer = model.make_optimizer(BASE_LR, ACT_LR_SCALE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    stopped_at = MAX_EPOCHS
    first_improvement_epoch = None
    baseline_val_loss = None
    best_path = os.path.join(BASE_DIR, f'_best_{act_name}_{norm_mode}_d{n_layers}_s{seed}.pt')
    epoch_rows, weight_norm_rows = ([], [])
    print(f'\n>>> [{device}] {act_name} | depth={n_layers} | seed={seed}')
    start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = (inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_acc = evaluate_acc(model, val_loader, device)
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
        if epoch in WEIGHT_NORM_EPOCHS:
            weight_norm_rows.extend([{**r, 'activation': act_name, 'seed': seed, 'n_layers': n_layers} for r in collect_weight_norms(model, epoch)])
        if epoch % 5 == 0 or epoch == 1:
            print(f'   [{device}] Epoch {epoch:02d}/{MAX_EPOCHS} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | patience {epochs_no_improve}/{PATIENCE}')
        if epochs_no_improve >= PATIENCE and epoch >= MIN_EPOCHS:
            stopped_at = epoch
            print(f'   [{device}] >>> Early stop at epoch {epoch}')
            break
    model.load_state_dict(torch.load(best_path))
    os.remove(best_path)
    test_acc = evaluate_acc(model, test_loader, device)
    converged = model.activation.get_param()
    print(f'   [{device}] FINAL {act_name} d{n_layers} s{seed} test_acc={test_acc:.4f} | stopped_at={stopped_at} | time={time.time() - start:.1f}s')
    criterion_diag = nn.CrossEntropyLoss()
    diag_inputs = DIAG_INPUTS_CPU.to(device)
    diag_labels = DIAG_LABELS_CPU.to(device)
    layer_stats_eval = collect_layer_stats(model, diag_inputs, diag_labels, criterion_diag, device, bn_mode='eval')
    layer_stats_train = collect_layer_stats(model, diag_inputs, diag_labels, criterion_diag, device, bn_mode='train_batch')
    layer_stats = layer_stats_eval + layer_stats_train
    for row in layer_stats:
        row.update({'activation': act_name, 'norm_mode': norm_mode, 'n_layers': n_layers, 'seed': seed})
    weight_norm_rows.extend([{**r, 'activation': act_name, 'seed': seed, 'n_layers': n_layers, 'final': True} for r in collect_weight_norms(model, epoch='final')])
    return (test_acc, converged, layer_stats, weight_norm_rows, stopped_at, first_improvement_epoch, epoch_rows)
_csv_lock = threading.Lock()

def run_depth_grid(depth, device_str, activations, seeds):
    device = torch.device(device_str)
    tag = f'd{depth}'
    results_path = os.path.join(BASE_DIR, f'CIFAR_C1RhoSlope__results_{tag}.csv')
    diag_path = os.path.join(BASE_DIR, f'CIFAR_C1RhoSlope__diagnostics_{tag}.csv')
    epochs_path = os.path.join(BASE_DIR, f'CIFAR_C1RhoSlope__epochs_{tag}.csv')
    wnorm_path = os.path.join(BASE_DIR, f'CIFAR_C1RhoSlope__weightnorms_{tag}.csv')
    all_results, all_diag, all_epochs, all_wnorm = ([], [], [], [])
    for path, store in [(results_path, 'results'), (diag_path, 'diag'), (epochs_path, 'epochs'), (wnorm_path, 'wnorm')]:
        if os.path.exists(path):
            try:
                rows = pd.read_csv(path).to_dict('records')
                if store == 'results':
                    all_results = rows
                elif store == 'diag':
                    all_diag = rows
                elif store == 'epochs':
                    all_epochs = rows
                elif store == 'wnorm':
                    all_wnorm = rows
            except Exception as e:
                print(f'[{tag}] could not load {path}: {e}')
    completed = {(str(r['activation']), int(r['seed'])) for r in all_results}
    run_specs = [(act, s) for act in activations for s in seeds if (act, s) not in completed]
    print(f'[{tag}] {len(completed)} already done, {len(run_specs)} remaining')
    for act_name, seed in run_specs:
        act_class = ACTIVATIONS[act_name]
        test_acc, converged, layer_stats, wnorm_rows, stopped_at, first_improvement_epoch, epoch_rows = train_one_experiment(act_name, act_class, NORM_MODE, depth, seed, device)
        row = {'activation': act_name, 'norm_mode': NORM_MODE, 'n_layers': depth, 'seed': seed, 'test_acc': round(test_acc, 4), 'stopped_at': stopped_at, 'first_improvement_epoch': first_improvement_epoch}
        row.update({f'converged_{k}': v for k, v in converged.items()})
        all_results.append(row)
        all_diag.extend(layer_stats)
        all_epochs.extend(epoch_rows)
        all_wnorm.extend(wnorm_rows)
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        pd.DataFrame(all_diag).to_csv(diag_path, index=False)
        pd.DataFrame(all_epochs).to_csv(epochs_path, index=False)
        pd.DataFrame(all_wnorm).to_csv(wnorm_path, index=False)
        with _csv_lock:
            print(f'[{tag}] saved {act_name} seed={seed} ({len(all_results)}/{len(activations) * len(seeds)})')
    print(f'[{tag}] DONE.')

def run_depth_grid_safe(depth, device_str, activations, seeds, error_flags):
    tag = f'd{depth}'
    try:
        run_depth_grid(depth, device_str, activations, seeds)
    except Exception:
        print(f"\n{'!' * 60}\n[{tag}] THREAD CRASHED on {device_str} -- see traceback below\n{'!' * 60}", file=sys.stderr, flush=True)
        traceback.print_exc()
        error_flags[tag] = traceback.format_exc()
if __name__ == '__main__':
    devices = [f'cuda:{i % max(N_GPUS, 1)}' for i in range(len(CORE_DEPTHS))]
    error_flags = {}
    threads = []
    for depth, device_str in zip(CORE_DEPTHS, devices):
        t = threading.Thread(target=run_depth_grid_safe, args=(depth, device_str, CORE_ACTIVATIONS, CORE_SEEDS, error_flags), name=f'depth{depth}-{device_str}')
        t.start()
        threads.append(t)
    heartbeat_s = 300
    while any((t.is_alive() for t in threads)):
        for t in threads:
            t.join(timeout=heartbeat_s)
        alive = [t.name for t in threads if t.is_alive()]
        if alive:
            print(f'[heartbeat] still running: {alive}', flush=True)
    print('\n' + '=' * 60)
    print('ALL DEPTHS DONE' if not error_flags else f'FINISHED WITH ERRORS in: {list(error_flags.keys())}')
    print('=' * 60)
    for kind in ['results', 'diagnostics', 'epochs', 'weightnorms']:
        parts = []
        for depth in CORE_DEPTHS:
            p = os.path.join(BASE_DIR, f'CIFAR_C1RhoSlope__{kind}_d{depth}.csv')
            if os.path.exists(p):
                parts.append(pd.read_csv(p))
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            combined.to_csv(os.path.join(BASE_DIR, f'CIFAR_C1RhoSlope__{kind}_combined.csv'), index=False)
            print(f'Wrote CIFAR_C1RhoSlope__{kind}_combined.csv ({len(combined)} rows)')
