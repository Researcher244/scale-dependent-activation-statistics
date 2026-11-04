import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
NEW_FUNCTIONS_ONLY = True
NEW_ACTIVATIONS = ['Logish', 'Smish', 'Gish', 'S-Mish', 'CoLU']
CONFIG = {'architecture': 'CIFAR_ResNet32_optionA', 'n_layers': 32, 'norm_mode': 'batchnorm', 'blocks_per_stage': 5, 'widths': [16, 32, 64], 'activations': ['ScaledSwish', 'ReLU', 'SetD5_HighA_LowGamma', 'SetD1_LowA_HighGamma', 'SetD6_HighA_HighGamma', 'ScaledSwishhalf', 'Mish', 'Swish', 'MatchAllToMish', 'SerfAct', 'TanhExpAct', 'GeluAct', 'PhishAct', 'SquareSwishAct', 'ExpExpishAct', 'FixC1MatchSlopeAndRho', 'FixC1FixRhoMatchSlope', 'FixC1FixSlopeMatchRho', 'FixRhoMatchC1AndSlope', 'FixRhoFixSlopeMatchC1', 'FixSlopeMatchC1AndRho', 'FixSlopeandC1HighRho', 'FixSlopeandC1LowRho', 'BandHigh', 'BandLow', 'GentleTiltDown09', 'HighBandSwish', 'LowBandSwish', 'SetD2_LowA_LowGamma', 'SetD3_MidA_LowGamma', 'SetD4_MidA_HighGamma'], 'seeds': [42, 43, 44, 45, 46], 'batch_size': 256, 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True, 'max_epochs': 60, 'min_epochs': 30, 'patience': 15, 'min_delta': 0.0, 'schedule': 'cosine', 'amp': True, 'clip_grad': None, 'dropout': 0.0, 'zero_init_residual': False, 'conv_init': 'kaiming_normal_fan_in_relu', 'head_init': 'xavier_normal', 'bn_eps': 1e-05, 'bn_momentum': 0.1, 'split_seed': 42, 'validation_fraction': 1 / 6, 'mean': [0.4914, 0.4822, 0.4465], 'std': [0.247, 0.2435, 0.2616], 'train_drop_last': True, 'augmentation': 'pad4_crop32_hflip0.5', 'augmentation_rng': 'per_seed_epoch_original_index_v1', 'cudnn_benchmark': False, 'cudnn_deterministic': True, 'tf32': False, 'cpu_threads': 2, 'diagnostic_epochs': [0, 6, 12, 30], 'diagnostic_batch_size': 32}
ORIGINAL_ACTIVATIONS = list(CONFIG['activations'])
CONFIG['activations'] = NEW_ACTIVATIONS if NEW_FUNCTIONS_ONLY else ORIGINAL_ACTIVATIONS + NEW_ACTIVATIONS

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
ACTIVATIONS['ReLU'] = lambda: nn.ReLU(inplace=False)

class OptionA(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.pad = (out_channels - in_channels) // 2

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad, self.pad))

class BasicBlock(nn.Module):

    def __init__(self, in_channels, channels, stride, factory, cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=cfg['bn_eps'], momentum=cfg['bn_momentum'])
        self.act1 = factory()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=cfg['bn_eps'], momentum=cfg['bn_momentum'])
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else OptionA(in_channels, channels, stride)
        self.act2 = factory()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + self.shortcut(x))

class ResNet32(nn.Module):

    def __init__(self, activation, cfg=CONFIG):
        super().__init__()
        factory = ACTIVATIONS[activation]
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=cfg['bn_eps'], momentum=cfg['bn_momentum'])
        self.act = factory()
        blocks, in_c = ([], 16)
        for stage, c in enumerate(cfg['widths']):
            for j in range(cfg['blocks_per_stage']):
                stride = 2 if stage > 0 and j == 0 else 1
                blocks.append(BasicBlock(in_c, c, stride, factory, cfg))
                in_c = c
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(64, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        return self.output(F.adaptive_avg_pool2d(x, 1).flatten(1))

def stream_seed(seed, epoch, index=0):
    return (int(seed) * 1000000007 + int(epoch) * 1000003 + int(index) * 9176 + 12345) % (2 ** 63 - 1)

class AugmentedCIFAR(Dataset):

    def __init__(self, base, indices, cfg):
        self.base, self.indices, self.cfg = (base, np.asarray(indices), cfg)
        self.context = torch.zeros(2, dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, seed, epoch):
        self.context[0], self.context[1] = (seed, epoch)

    def __getitem__(self, k):
        from torchvision.transforms import functional as TF
        index = int(self.indices[k])
        image, label = self.base[index]
        gen = torch.Generator().manual_seed(stream_seed(self.context[0], self.context[1], index))
        image = TF.pad(image, 4, fill=0)
        top, left = torch.randint(0, 9, (2,), generator=gen).tolist()
        image = TF.crop(image, top, left, 32, 32)
        if torch.rand((), generator=gen).item() < 0.5:
            image = TF.hflip(image)
        image = TF.normalize(TF.to_tensor(image), self.cfg['mean'], self.cfg['std'])
        return (image, label)

def get_datasets(data_dir, cfg, download=False):
    from torchvision.datasets import CIFAR10
    from torchvision import transforms as T
    from sklearn.model_selection import StratifiedShuffleSplit
    base = CIFAR10(data_dir, train=True, download=download, transform=None)
    ev = T.Compose([T.ToTensor(), T.Normalize(cfg['mean'], cfg['std'])])
    base_eval = CIFAR10(data_dir, train=True, download=False, transform=ev)
    test = CIFAR10(data_dir, train=False, download=download, transform=ev)
    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=cfg['validation_fraction'], random_state=cfg['split_seed']).split(np.zeros(len(base.targets)), np.asarray(base.targets)))
    return (base, base_eval, test, train_idx, val_idx)

def make_training_loader(dataset, cfg, workers, seed):
    order_gen = torch.Generator().manual_seed(stream_seed(seed, 0))
    worker_gen = torch.Generator().manual_seed(stream_seed(seed, 0, 1))
    extra = {'multiprocessing_context': 'spawn', 'persistent_workers': True} if workers else {}
    return DataLoader(dataset, batch_size=cfg['batch_size'], sampler=RandomSampler(dataset, generator=order_gen), drop_last=cfg['train_drop_last'], num_workers=workers, pin_memory=True, generator=worker_gen, **extra)

def make_loaders(datasets, cfg, workers, seed):
    base, base_eval, test, ti, vi = datasets
    augmented = AugmentedCIFAR(base, ti, cfg)
    train = make_training_loader(augmented, cfg, workers, seed)
    val = DataLoader(Subset(base_eval, vi), batch_size=cfg['batch_size'], num_workers=0, pin_memory=True)
    test_loader = DataLoader(test, batch_size=cfg['batch_size'], num_workers=0, pin_memory=True)
    diag = torch.stack([base_eval[int(i)][0] for i in ti[:cfg['diagnostic_batch_size']]])
    return (train, val, test_loader, diag)

def seed_all(seed, cfg):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = cfg['cudnn_benchmark']
    torch.backends.cudnn.deterministic = cfg['cudnn_deterministic']
    torch.backends.cuda.matmul.allow_tf32 = cfg['tf32']
    torch.backends.cudnn.allow_tf32 = cfg['tf32']

def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def finite_state(model):
    ts = [v for v in model.state_dict().values() if v.is_floating_point()]
    return bool(torch.stack([torch.isfinite(v).all() for v in ts]).all().item())

def atomic_torch(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, float) and (not math.isfinite(obj)):
        return None
    return obj

def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(clean_json(obj), indent=2, allow_nan=False))
    os.replace(tmp, path)

def atomic_csv(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)

class PauseRun(Exception):
    pass

def check_pause(stop_path):
    if stop_path is not None and Path(stop_path).exists():
        raise PauseRun('Stop requested; resume from the last complete epoch.')

def amp_context(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)

def evaluate(model, loader, device, amp, stop_path=None):
    model.eval()
    total_loss, correct, n = (0.0, 0, 0)
    with torch.no_grad():
        for inputs, labels in loader:
            check_pause(stop_path)
            inputs, labels = (inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True))
            with amp_context(device, amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(logits).all().item()) or not math.isfinite(loss.item()):
                return (float('nan'), float('nan'), 'nonfinite_evaluation_logits_or_loss')
            size = labels.numel()
            total_loss += loss.item() * size
            correct += (logits.argmax(1) == labels).sum().item()
            n += size
    if not n:
        raise ValueError('Empty evaluation loader')
    return (total_loss / n, correct / n, '')

def diagnostic_snapshot(model, inputs, device, amp, epoch):
    if inputs is None:
        return []
    was_training = model.training
    rows, handles = ([], [])
    sites = [('stem', model.act)]
    for i, block in enumerate(model.blocks):
        sites += [(f'block{i + 1}.branch', block.act1), (f'block{i + 1}.post_add', block.act2)]

    def hook(name):

        def capture(module, args, out):
            z, y = (args[0].detach().float(), out.detach().float())
            rows.append({'epoch': epoch, 'site': name, 'mode': 'eval', 'input_mean': z.mean().item(), 'input_std': z.std(unbiased=False).item(), 'input_max_abs': z.abs().max().item(), 'output_mean': y.mean().item(), 'output_std': y.std(unbiased=False).item(), 'finite': bool(torch.isfinite(z).all().item() and torch.isfinite(y).all().item())})
        return capture
    try:
        model.eval()
        for name, module in sites:
            handles.append(module.register_forward_hook(hook(name)))
        with torch.no_grad(), amp_context(device, amp):
            model(inputs.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return rows

def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}

def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)

def stop_update(epoch, loss, best, bad, cfg):
    improved = loss < best - cfg['min_delta']
    bad = 0 if improved else bad + 1
    if epoch == cfg['min_epochs']:
        bad = 0
    stop = epoch >= cfg['min_epochs'] and bad >= cfg['patience']
    return (improved, bad, stop)

def run_one(activation, seed, device, loaders, out, cfg=CONFIG, stop_path=None, pause_after_epoch=None, model_factory=ResNet32):
    run_id = f'{activation}__s{seed}'
    rd = Path(out) / 'runs' / run_id
    rd.mkdir(parents=True, exist_ok=True)
    result_path, cp_path = (rd / 'result.json', rd / 'resume.pt')
    if result_path.exists():
        return json.loads(result_path.read_text())
    train, val, test, diag_inputs = loaders
    amp = cfg['amp'] and device.type == 'cuda'
    seed_all(seed, cfg)
    model = model_factory(activation, cfg).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'], nesterov=cfg['nesterov'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    cp = {'epoch': 0, 'best_loss': float('inf'), 'best_epoch': None, 'best_state': None, 'bad': 0, 'epochs': [], 'diagnostics': [], 'phase': 'training', 'initial_state_sha256': state_hash(model.state_dict()), 'train_val_seconds': 0.0, 'failure': '', 'failure_epoch': None, 'failure_batch': None, 'training_status': ''}

    def save():
        cp.update(model=cpu_state(model), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(device))
        atomic_torch(cp_path, cp)
        atomic_csv(rd / 'epochs.csv', cp['epochs'])
        atomic_csv(rd / 'diagnostics.csv', cp['diagnostics'])
        atomic_json(rd / 'progress.json', {'activation': activation, 'seed': seed, 'epoch': cp['epoch'], 'phase': cp['phase'], 'initial_state_sha256': cp['initial_state_sha256']})
    if cp_path.exists():
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler'])
        scaler.load_state_dict(cp['scaler'])
        restore_rng(cp['rng'], device)
        atomic_csv(rd / 'epochs.csv', cp['epochs'])
        atomic_csv(rd / 'diagnostics.csv', cp['diagnostics'])
        print(f"[{device}] RESUME {run_id} after epoch {cp['epoch']}", flush=True)
    else:
        if 0 in cfg['diagnostic_epochs']:
            cp['diagnostics'] += diagnostic_snapshot(model, diag_inputs, device, amp, 0)
        save()
    print(f'[{device}] {run_id} | {sum((p.numel() for p in model.parameters())):,} parameters', flush=True)
    if cp['phase'] == 'training':
        for epoch in range(cp['epoch'] + 1, cfg['max_epochs'] + 1):
            check_pause(stop_path)
            epoch_start = time.monotonic()
            seed_all(stream_seed(seed, epoch), cfg)
            if hasattr(train.dataset, 'set_epoch'):
                train.dataset.set_epoch(seed, epoch)
            if train.generator is not None:
                train.generator.manual_seed(stream_seed(seed, epoch))
            if getattr(train.sampler, 'generator', None) is not None:
                train.sampler.generator.manual_seed(stream_seed(seed, epoch))
            model.train()
            loss_sum, correct, count, skips, max_grad = (0.0, 0, 0, 0, 0.0)
            failure, failure_batch = ('', None)
            lr = optimizer.param_groups[0]['lr']
            for batch, (inputs, labels) in enumerate(train, 1):
                check_pause(stop_path)
                inputs, labels = (inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True))
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, amp):
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, labels)
                if not bool(torch.isfinite(logits).all().item()) or not math.isfinite(loss.item()):
                    failure = 'training_logits_or_loss' if finite_state(model) else 'parameters_or_bn_buffers'
                    failure_batch = batch
                    break
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_max = torch.stack([p.grad.detach().abs().max() for p in model.parameters() if p.grad is not None]).max().item()
                max_grad = max(max_grad, grad_max) if math.isfinite(grad_max) else float('inf')
                if not math.isfinite(grad_max) and (not amp):
                    failure, failure_batch = ('nonfinite_gradient_without_AMP', batch)
                    break
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skips += int(scaler.get_scale() < previous_scale)
                loss_sum += loss.item() * labels.numel()
                correct += (logits.argmax(1) == labels).sum().item()
                count += labels.numel()
            if not failure and (not finite_state(model)):
                failure = 'parameters_or_bn_buffers'
            if failure:
                vl, va = (float('nan'), float('nan'))
            else:
                vl, va, failure = evaluate(model, val, device, amp, stop_path)
                if failure:
                    failure = 'validation_' + failure
            if not count and (not failure):
                raise ValueError('No training examples; check batch size/drop_last')
            elapsed = time.monotonic() - epoch_start
            row = {'run_id': run_id, 'activation': activation, 'norm_mode': cfg['norm_mode'], 'architecture': cfg['architecture'], 'n_layers': cfg['n_layers'], 'seed': seed, 'epoch': epoch, 'lr': lr, 'train_loss': loss_sum / count if count else float('nan'), 'train_acc': correct / count if count else float('nan'), 'train_examples': count, 'val_loss': vl, 'val_acc': va, 'max_abs_gradient': max_grad, 'amp_skipped_steps': skips, 'failure': failure, 'failure_batch': failure_batch, 'epoch_train_val_seconds': elapsed, 'epoch_complete': not bool(failure)}
            cp['epochs'].append(row)
            cp['epoch'] = epoch
            cp['train_val_seconds'] += elapsed
            if failure:
                cp.update(phase='trained', training_status='diverged', failure=failure, failure_epoch=epoch, failure_batch=failure_batch)
            else:
                improved, cp['bad'], stopped = stop_update(epoch, vl, cp['best_loss'], cp['bad'], cfg)
                if improved:
                    cp.update(best_loss=vl, best_epoch=epoch, best_state=cpu_state(model))
                scheduler.step()
                if epoch in cfg['diagnostic_epochs']:
                    cp['diagnostics'] += diagnostic_snapshot(model, diag_inputs, device, amp, epoch)
                if stopped or epoch == cfg['max_epochs']:
                    cp.update(phase='trained', training_status='early_stopped' if stopped else 'budget_complete')
            save()
            print(f"[{device}] {run_id} epoch {epoch:02d}/{cfg['max_epochs']}: train={row['train_loss']:.5f} val={vl:.5f} acc={va:.4f} patience={cp['bad']}/{cfg['patience']} AMP_skips={skips}" + (f' FAILURE={failure}' if failure else ''), flush=True)
            if pause_after_epoch == epoch:
                raise PauseRun('Synthetic resume test')
            if cp['phase'] == 'trained':
                break
    check_pause(stop_path)
    test_loss, test_acc, test_failure = (float('nan'), float('nan'), '')
    if cp['best_state'] is not None:
        model.load_state_dict(cp['best_state'])
        test_loss, test_acc, test_failure = evaluate(model, test, device, amp, stop_path)
        end_diags = diagnostic_snapshot(model, diag_inputs, device, amp, 'best')
        cp['diagnostics'] = [r for r in cp['diagnostics'] if r['epoch'] != 'best'] + end_diags
        atomic_csv(rd / 'diagnostics.csv', cp['diagnostics'])
        atomic_torch(rd / 'best.pt', {'state_dict': cp['best_state'], 'activation': activation, 'seed': seed, 'epoch': cp['best_epoch'], 'config': cfg})
    diverged = cp['training_status'] == 'diverged'
    status = 'diverged' if diverged else 'evaluation_failed' if test_failure else cp['training_status']
    result = {'run_id': run_id, 'activation': activation, 'architecture': cfg['architecture'], 'norm_mode': cfg['norm_mode'], 'n_layers': cfg['n_layers'], 'seed': seed, 'status': status, 'diverged': diverged, 'diverged_epoch': cp['failure_epoch'], 'failure': cp['failure'] or test_failure, 'failure_batch': cp['failure_batch'], 'stopped_at': cp['epoch'], 'best_epoch': cp['best_epoch'], 'best_val_loss': cp['best_loss'], 'test_loss': test_loss if not diverged else float('nan'), 'test_acc': test_acc if not diverged else float('nan'), 'best_finite_checkpoint_test_loss': test_loss, 'best_finite_checkpoint_test_acc': test_acc, 'train_val_seconds': cp['train_val_seconds'], 'initial_state_sha256': cp['initial_state_sha256']}
    atomic_json(result_path, result)
    print(f"[{device}] FINAL {run_id}: {status}, test_acc={result['test_acc']:.4f}", flush=True)
    return result

def merge_outputs(out, cfg=CONFIG):
    out = Path(out)
    results, epochs, diagnostics, statuses = ([], [], [], [])
    for seed in cfg['seeds']:
        for a in cfg['activations']:
            rd = out / 'runs' / f'{a}__s{seed}'
            result = json.loads((rd / 'result.json').read_text()) if (rd / 'result.json').exists() else None
            if result:
                results.append(result)
            progress = json.loads((rd / 'progress.json').read_text()) if (rd / 'progress.json').exists() else {}
            statuses.append({'activation': a, 'seed': seed, 'status': result['status'] if result else 'incomplete' if progress else 'pending', 'last_epoch': progress.get('epoch', 0)})
            for filename, target in [('epochs.csv', epochs), ('diagnostics.csv', diagnostics)]:
                p = rd / filename
                if p.exists():
                    try:
                        records = pd.read_csv(p).to_dict('records')
                    except pd.errors.EmptyDataError:
                        records = []
                    for r in records:
                        r.update(activation=a, seed=seed)
                    target.extend(records)
    for name, rows in [('results', results), ('epochs', epochs), ('diagnostics', diagnostics), ('run_status', statuses)]:
        atomic_csv(out / f'{name}.csv', rows)
    summary = []
    for a in cfg['activations']:
        rs = [r for r in results if r['activation'] == a]
        ok = [r for r in rs if r['status'] in ['budget_complete', 'early_stopped'] and r['test_acc'] is not None]
        xs = [r['test_acc'] for r in ok]
        summary.append({'activation': a, 'planned_seeds': len(cfg['seeds']), 'finished': len(rs), 'numerical_failures': sum((r['diverged'] for r in rs)), 'evaluation_failures': sum((r['status'] == 'evaluation_failed' for r in rs)), 'finite_test_results': len(xs), 'pending_or_incomplete': len(cfg['seeds']) - len(rs), 'test_acc_mean_finite': np.mean(xs) if xs else np.nan, 'test_acc_sd_finite': np.std(xs, ddof=1) if len(xs) > 1 else np.nan})
    atomic_csv(out / 'summary.csv', summary)
    return statuses
LEGACY_SCRIPT_SHA256 = 'ba2b45c245c312bfce0a504a618a1fd8c9b75062dc160f80e0a36146234857df'
LEGACY_ACTIVATIONS = ['ScaledSwish', 'ReLU', 'SetD5_HighA_LowGamma', 'SetD1_LowA_HighGamma', 'SetD6_HighA_HighGamma', 'ScaledSwishhalf', 'Mish', 'Swish', 'MatchAllToMish']

def load_or_create_manifest(out, datasets, workers, gpu_names, cfg=CONFIG):
    import torchvision
    import sklearn
    _, _, _, ti, vi = datasets
    data_root = Path(datasets[0].root) / 'cifar-10-batches-py'
    content_hash = hashlib.sha256()
    for f in sorted(data_root.glob('*')):
        if f.is_file():
            content_hash.update(f.name.encode())
            with f.open('rb') as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b''):
                    content_hash.update(block)
    manifest = {'config': cfg, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'dataset_sha256': content_hash.hexdigest(), 'split_sha256': hashlib.sha256(np.asarray(ti, dtype=np.int64).tobytes() + np.asarray(vi, dtype=np.int64).tobytes()).hexdigest(), 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'python': sys.version.split()[0], 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'device_types': sorted(set(gpu_names)), 'num_workers': workers}
    path = Path(out) / 'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != manifest:
            keys = [k for k in set(previous) | set(manifest) if previous.get(k) != manifest.get(k)]
            old_cfg, new_cfg = (previous.get('config', {}), manifest['config'])
            old_acts, new_acts = (old_cfg.get('activations', []), new_cfg['activations'])
            panel_expansion = set(keys) <= {'config', 'script_sha256'} and previous.get('script_sha256') == LEGACY_SCRIPT_SHA256 and ({k: v for k, v in old_cfg.items() if k != 'activations'} == {k: v for k, v in new_cfg.items() if k != 'activations'}) and (old_acts == LEGACY_ACTIVATIONS) and (new_acts[:len(old_acts)] == old_acts) and (len(new_acts) == len(set(new_acts))) and (len(new_acts) > len(old_acts))
            if not panel_expansion:
                raise RuntimeError(f'Protocol/environment changed ({keys}). Restore the original configuration or use a NEW output directory.')
            backup = Path(out) / 'manifest.before_panel_expansion.json'
            if not backup.exists():
                atomic_json(backup, previous)
            atomic_json(path, manifest)
            print(f'Expanded activation panel: {len(old_acts)} -> {len(new_acts)}. Existing completed runs will be skipped.', flush=True)
    else:
        atomic_json(path, manifest)
        np.savez(Path(out) / 'split_indices.npz', train=ti, val=vi)
    return manifest
SHARD_ASSIGNMENT = [['Logish', 'Smish', 'Gish'], ['S-Mish', 'CoLU']]

def shard_activations(shard, n_shards):
    acts = CONFIG['activations']
    assigned = [a for group in SHARD_ASSIGNMENT for a in group]
    if n_shards == len(SHARD_ASSIGNMENT) and sorted(assigned) == sorted(acts):
        return list(SHARD_ASSIGNMENT[shard])
    return acts[shard::n_shards]

def worker(args):
    torch.set_num_threads(CONFIG['cpu_threads'])
    device = torch.device(f'cuda:{args.shard}')
    torch.cuda.set_device(device)
    datasets = get_datasets(args.data, CONFIG)
    acts = shard_activations(args.shard, args.shards)
    try:
        for seed in CONFIG['seeds']:
            for a in acts:
                check_pause(Path(args.out) / 'STOP')
                if (Path(args.out) / 'runs' / f'{a}__s{seed}' / 'result.json').exists():
                    continue
                loaders = make_loaders(datasets, CONFIG, args.workers, seed)
                try:
                    run_one(a, seed, device, loaders, args.out, stop_path=Path(args.out) / 'STOP')
                finally:
                    iterator = getattr(loaders[0], '_iterator', None)
                    if iterator is not None:
                        iterator._shutdown_workers()
                    del loaders
                    torch.cuda.empty_cache()
    except PauseRun as exc:
        print(f'[gpu{args.shard}] PAUSED: {exc}', flush=True)
    except Exception:
        atomic_json(Path(args.out) / f'worker_error_g{args.shard}.json', {'traceback': traceback.format_exc(), 'time': time.time()})
        raise

def parent(args):
    import fcntl
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    lock = (out / 'experiment.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This experiment is already running. Stop its existing process before restarting.')
    n = min(args.gpus, torch.cuda.device_count())
    if n < 1:
        raise RuntimeError('Enable a Kaggle GPU accelerator; this training launcher requires CUDA.')
    datasets = get_datasets(args.data, CONFIG, download=True)
    load_or_create_manifest(out, datasets, args.workers, [torch.cuda.get_device_name(i) for i in range(n)])
    del datasets
    (out / 'STOP').unlink(missing_ok=True)
    for p in out.glob('worker_error_g*.json'):
        p.unlink()
    atomic_json(out / 'run_plan.json', {'runs': len(CONFIG['activations']) * len(CONFIG['seeds']), 'gpu_shards': [shard_activations(i, n) for i in range(n)]})
    print(f"PLAN: {len(CONFIG['activations'])} activations x {len(CONFIG['seeds'])} seeds = {len(CONFIG['activations']) * len(CONFIG['seeds'])} runs, {n} GPU processes. Output: {out}", flush=True)
    for i in range(n):
        print(f'Shard g{i} -> cuda:{i}: {shard_activations(i, n)}', flush=True)
    processes = []
    statuses = merge_outputs(out)
    started, last_report, last_merge = (time.monotonic(), 0.0, 0.0)
    requested_stop = [False]

    def stop_handler(signum, frame):
        requested_stop[0] = True
        (out / 'STOP').touch()
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    stop_started = None
    try:
        for shard in range(n):
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--shard', str(shard), '--shards', str(n), '--out', str(out), '--data', str(Path(args.data).resolve()), '--workers', str(args.workers)]
            env = dict(os.environ, OMP_NUM_THREADS=str(CONFIG['cpu_threads']), MKL_NUM_THREADS=str(CONFIG['cpu_threads']), PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
            processes.append(subprocess.Popen(cmd, env=env, pass_fds=(lock.fileno(),)))
        while any((p.poll() is None for p in processes)):
            now = time.monotonic()
            if now - last_merge >= 30:
                statuses = merge_outputs(out)
                last_merge = now
            if now - last_report >= 60:
                done = sum((r['status'] not in ('pending', 'incomplete') for r in statuses))
                print(f"[heartbeat] {done}/{len(CONFIG['activations']) * len(CONFIG['seeds'])} runs finalized; session elapsed {(now - started) / 3600:.2f} h", flush=True)
                last_report = now
            if any((p.poll() not in (None, 0) for p in processes)):
                requested_stop[0] = True
            if args.session_hours > 0 and now - started >= args.session_hours * 3600:
                requested_stop[0] = True
            if requested_stop[0] or (out / 'STOP').exists():
                (out / 'STOP').touch()
                if stop_started is None:
                    stop_started = now
                    print('Pausing workers. An interrupted epoch will replay on resume.', flush=True)
                if now - stop_started > 60:
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
            time.sleep(1)
    finally:
        (out / 'STOP').touch()
        for p in processes:
            if p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        merge_outputs(out)
    if any((p.returncode not in (0, -signal.SIGTERM) for p in processes)):
        raise RuntimeError('A worker failed; inspect worker_error_g*.json. Valid checkpoints remain resumable.')
    print('Saved results.csv, epochs.csv, diagnostics.csv, summary.csv and run_status.csv.', flush=True)
    print('Preserve the ENTIRE output folder, including runs/*/resume.pt, when moving Kaggle sessions.', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='CIFAR_ResNet32_NewFunctions')
    parser.add_argument('--data', default='data')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--session-hours', type=float, default=0.0, help='Pause before typical Kaggle session expiry; 0 disables this limit.')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 0 or args.gpus < 1:
        parser.error('workers >= 0 and gpus >= 1 required')
    if args.merge_only:
        merge_outputs(args.out)
    elif args.worker:
        worker(args)
    else:
        parent(args)
if __name__ == '__main__':
    main()

