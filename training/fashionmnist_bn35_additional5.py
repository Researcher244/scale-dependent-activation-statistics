"""Fashion-MNIST BN35; five additional functions; five seeds.

Five seeds per activation; source training and resume rules retained.
"""

from pathlib import Path
import csv, hashlib, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
DEPTH = 35
BASE_DIR = __import__('pathlib').Path(__file__).resolve().parents[1] / 'runs/fashionmnist_bn35_additional5'
__import__('pathlib').Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
DATA_DIR = __import__('pathlib').Path(__file__).resolve().parents[1] / 'datasets'
OUTPUT_DIR = __import__('pathlib').Path(__file__).resolve().parents[1] / 'runs/fashionmnist_bn35_additional5'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 43, 44, 45, 46]
WIDTH = 500
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = {35: (100, 50, 20), 10: (20, 10, 5)}[DEPTH]

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

def make_norm_layer(mode, num_features):
    if mode != 'batchnorm':
        raise ValueError('This experiment uses batch normalization.')
    return nn.BatchNorm1d(num_features, affine=True)

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
VERSION = 'FashionMNIST_new_literature_panel_1.0'
IMPLEMENTATION_SHA256 = '7338936eb69586f7cd6af3e01ff36a9325fc53a550d953ff58b9d1eecd9b5433'
PANEL = ['Logish', 'Smish', 'Gish', 'S-Mish', 'CoLU']
LITERATURE = {'Logish', 'Smish', 'Gish', 'S-Mish', 'CoLU', 'Mish', 'Swish', 'GeluAct', 'SerfAct', 'TanhExpAct', 'PhishAct', 'SquareSwishAct', 'ExpExpishAct'}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def configure_torch():
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)

def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean_json(value), indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)

def atomic_csv(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    columns = list(dict.fromkeys((k for row in rows for k in row)))
    with temp.open('w', newline='', encoding='utf-8') as file:
        if columns:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temp, path)

def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def state_hash(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class FashionData:

    def __init__(self, data_dir, out):
        import torchvision
        from sklearn.model_selection import StratifiedShuffleSplit
        full = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True)
        labels = full.targets.numpy()
        split = StratifiedShuffleSplit(n_splits=1, test_size=1 / 6, random_state=42)
        ti, vi = next(split.split(np.zeros(len(labels)), labels))
        saved = out / 'split_indices.npz'
        if saved.exists():
            with np.load(saved) as z:
                if not (np.array_equal(ti, z['train']) and np.array_equal(vi, z['validation'])):
                    raise RuntimeError('Stored data split differs; use a new output folder.')
        else:
            np.savez(saved, train=ti, validation=vi)
        total = total2 = 0.0
        count = 0
        for ix in np.array_split(ti, 50):
            values = full.data[ix].double().div_(255.0)
            total += values.sum().item()
            total2 += values.square().sum().item()
            count += values.numel()
        mean = total / count
        std = math.sqrt(max(total2 / count - mean * mean, 0.0))
        if not std > 0:
            raise RuntimeError('Training pixel standard deviation is zero.')
        h = hashlib.sha256()
        for t in (full.data, full.targets, test.data, test.targets):
            h.update(t.numpy().tobytes())
        self.info = {'train_count': len(ti), 'validation_count': len(vi), 'test_count': len(test), 'pixel_mean': mean, 'pixel_std': std, 'dataset_sha256': h.hexdigest(), 'split_sha256': hashlib.sha256(ti.tobytes() + vi.tobytes()).hexdigest()}
        images = full.data.float().div_(255.0).sub_(mean).div_(std).flatten(1)
        test_images = test.data.float().div_(255.0).sub_(mean).div_(std).flatten(1)
        self.sets = {'train': TensorDataset(images[ti], full.targets[ti]), 'validation': TensorDataset(images[vi], full.targets[vi]), 'test': TensorDataset(test_images, test.targets)}
        print(f'Fashion-MNIST: train={len(ti)}, validation={len(vi)}, test={len(test)}; train-only pixel mean={mean:.6f}, SD={std:.6f}', flush=True)

    def loader(self, part, seed, epoch=0):
        g = torch.Generator().manual_seed(seed + 100003 * epoch + 17)
        return DataLoader(self.sets[part], batch_size=BATCH_SIZE, shuffle=part == 'train', generator=g, num_workers=0, pin_memory=str(DEVICE).startswith('cuda'), drop_last=False)

class NonFinite(RuntimeError):
    pass

def train_epoch(model, optimizer, loader, device):
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0
    for batch, (inputs, labels) in enumerate(loader, 1):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss).item():
            raise NonFinite(f'non-finite training loss at batch {batch}')
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach() * len(labels)
        correct += (logits.detach().argmax(1) == labels).sum()
        count += len(labels)
    return (loss_sum.item() / count, correct.item() / count)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    count = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += F.cross_entropy(logits, labels, reduction='sum')
        correct += (logits.argmax(1) == labels).sum()
        count += len(labels)
    loss = loss_sum.item() / count
    if not math.isfinite(loss):
        raise NonFinite('non-finite evaluation loss')
    return (loss, correct.item() / count)

def state_is_finite(model):
    flags = [torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()]
    return bool(torch.stack(flags).all().item())

def stop_update(epoch, loss, best, stale):
    improved = loss < best
    stale = 0 if improved else stale + 1
    if epoch == MIN_EPOCHS:
        stale = 0
    return (improved, stale, epoch >= MIN_EPOCHS and stale >= PATIENCE)

def run_one(name, seed, data, out, device, pause_after_epoch=None):
    rid = f'{name}__s{seed}'
    folder = out / 'runs' / rid
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    if result_path.exists():
        print(f'SKIP {rid}: already finalized', flush=True)
        return json.loads(result_path.read_text())
    seed_all(seed)
    model = DeepMLP(ACTIVATIONS[name], DEPTH, 'batchnorm', WIDTH).to(device)
    optimizer = model.make_optimizer(LEARNING_RATE, 1.0)
    path = folder / 'resume.pt'
    if path.exists():
        cp = load_checkpoint(path)
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        print(f"RESUME {rid}: after epoch {cp['epoch']}", flush=True)
    else:
        cp = {'epoch': 0, 'best_loss': float('inf'), 'best_epoch': None, 'best_file': None, 'stale': 0, 'history': [], 'phase': 'training', 'training_status': '', 'elapsed_seconds': 0.0, 'initial_state_sha256': state_hash(model.state_dict())}
    base = {'run_id': rid, 'activation': name, 'group': 'literature' if name in LITERATURE else 'constructed', 'dataset': 'FashionMNIST', 'norm_mode': 'batchnorm', 'n_layers': DEPTH, 'width': WIDTH, 'seed': seed}
    atomic_csv(folder / 'epochs.csv', cp['history'])
    merge_epochs(out, rid, cp['history'])
    print(f'>>> {name} | BN | depth={DEPTH} | seed={seed} | device={device}', flush=True)
    failure, failure_epoch = ('', None)
    while cp['phase'] == 'training' and cp['epoch'] < MAX_EPOCHS:
        epoch = cp['epoch'] + 1
        started = time.perf_counter()
        seed_all(seed + 1000003 * epoch)
        try:
            train_loss, train_acc = train_epoch(model, optimizer, data.loader('train', seed, epoch), device)
            if not state_is_finite(model):
                raise NonFinite('non-finite weights or BN buffers after training')
            val_loss, val_acc = evaluate(model, data.loader('validation', seed), device)
        except NonFinite as exc:
            failure, failure_epoch = (str(exc), epoch)
            elapsed = time.perf_counter() - started
            cp['elapsed_seconds'] += elapsed
            row = dict(base, epoch=epoch, train_loss=float('nan'), train_acc=float('nan'), val_loss=float('nan'), val_acc=float('nan'), epoch_complete=False, failure=failure, best_epoch_so_far=cp['best_epoch'], epoch_train_val_seconds=elapsed)
            cp['history'].append(row)
            atomic_csv(folder / 'epochs.csv', cp['history'])
            merge_epochs(out, rid, cp['history'])
            print(f'>>> DIVERGED {rid} at epoch {epoch}: {failure}', flush=True)
            break
        elapsed = time.perf_counter() - started
        improved, stale, stop = stop_update(epoch, val_loss, cp['best_loss'], cp['stale'])
        weights = cpu_state(model)
        if improved:
            best_file = f'best_epoch_{epoch:03d}.pt'
            atomic_torch(folder / best_file, {'state_dict': weights, 'epoch': epoch})
            cp.update(best_loss=val_loss, best_epoch=epoch, best_file=best_file)
        cp.update(epoch=epoch, stale=stale)
        cp['elapsed_seconds'] += elapsed
        cp['history'].append(dict(base, epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, epoch_complete=True, failure='', best_epoch_so_far=cp['best_epoch'], epoch_train_val_seconds=elapsed))
        if stop or epoch == MAX_EPOCHS:
            cp['phase'] = 'trained'
            cp['training_status'] = 'early_stopped' if stop and epoch < MAX_EPOCHS else 'budget_complete'
        cp.update(model=weights, optimizer=optimizer.state_dict())
        atomic_torch(path, cp)
        atomic_csv(folder / 'epochs.csv', cp['history'])
        for old_best in folder.glob('best_epoch_*.pt'):
            if old_best.name != cp['best_file']:
                old_best.unlink()
        save_seconds = time.perf_counter() - started - elapsed
        print(f'{rid} epoch {epoch:3}/{MAX_EPOCHS}: train={train_loss:.5f} train_acc={train_acc:.4f} val={val_loss:.5f} acc={val_acc:.4f} patience={stale}/{PATIENCE} | train+val={elapsed:.1f}s save={save_seconds:.1f}s', flush=True)
        merge_epochs(out, rid, cp['history'])
        if pause_after_epoch == epoch:
            raise KeyboardInterrupt('Synthetic pause for resume verification')
    test_loss = test_acc = float('nan')
    status = 'diverged' if failure else cp['training_status']
    if not failure and cp['best_file']:
        best = load_checkpoint(folder / cp['best_file'])
        model.load_state_dict(best['state_dict'])
        try:
            test_loss, test_acc = evaluate(model, data.loader('test', seed), device)
        except NonFinite as exc:
            status, failure = ('evaluation_failed', str(exc))
    result = dict(base, status=status, diverged=status == 'diverged', failure=failure, failure_epoch=failure_epoch, stopped_at=failure_epoch or cp['epoch'], best_epoch=cp['best_epoch'], best_val_loss=cp['best_loss'], test_loss=test_loss, test_acc=test_acc, train_val_seconds=cp['elapsed_seconds'], initial_state_sha256=cp['initial_state_sha256'])
    atomic_json(result_path, result)
    merge_outputs(out)
    print(f"FINAL {rid}: {status} | best_epoch={cp['best_epoch']} test_loss={test_loss:.5f} test_acc={test_acc:.4f}", flush=True)
    return result

def read_csv_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))
_EPOCH_CACHE = {}

def merge_epochs(out, updated_run=None, updated_rows=None):
    key = str(Path(out).resolve())
    if key not in _EPOCH_CACHE:
        _EPOCH_CACHE[key] = {f'{a}__s{seed}': read_csv_rows(out / 'runs' / f'{a}__s{seed}' / 'epochs.csv') for seed in SEEDS for a in PANEL}
    cache = _EPOCH_CACHE[key]
    if updated_run is not None:
        cache[updated_run] = list(updated_rows)
    rows = [r for seed in SEEDS for a in PANEL for r in cache.get(f'{a}__s{seed}', [])]
    atomic_csv(out / 'epochs.csv', rows)

def merge_outputs(out):
    merge_epochs(out)
    results, statuses = ([], [])
    for seed in SEEDS:
        for a in PANEL:
            folder = out / 'runs' / f'{a}__s{seed}'
            path = folder / 'result.json'
            if path.exists():
                result = json.loads(path.read_text())
                results.append(result)
                status = result['status']
            else:
                status = 'incomplete' if (folder / 'resume.pt').exists() else 'pending'
            statuses.append({'activation': a, 'seed': seed, 'status': status})
    atomic_csv(out / 'results.csv', results)
    atomic_csv(out / 'run_status.csv', statuses)
    summary = []
    for a in PANEL:
        rs = [r for r in results if r['activation'] == a]
        xs = [r['test_acc'] for r in rs if r['status'] in ('budget_complete', 'early_stopped') and r.get('test_acc') is not None and math.isfinite(r['test_acc'])]
        summary.append({'activation': a, 'finished': len(rs), 'planned': len(SEEDS), 'diverged': sum((r['status'] == 'diverged' for r in rs)), 'finite_test_results': len(xs), 'test_acc_mean_finite': float(np.mean(xs)) if xs else None, 'test_acc_sd_finite': float(np.std(xs, ddof=1)) if len(xs) > 1 else None})
    atomic_csv(out / 'summary.csv', summary)
    return len(results)

def ensure_manifest(out, data, device):
    import torchvision, sklearn
    manifest = {'version': VERSION, 'implementation_sha256': IMPLEMENTATION_SHA256, 'dataset': 'FashionMNIST', 'data': data.info, 'depth': DEPTH, 'width': WIDTH, 'dropout': 0.25, 'normalization': 'batchnorm', 'activations': PANEL, 'seeds': SEEDS, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'optimizer': 'SGD', 'momentum': 0.0, 'weight_decay': 0.0, 'max_epochs': MAX_EPOCHS, 'min_epochs': MIN_EPOCHS, 'patience': PATIENCE, 'min_delta': 0.0, 'scheduler': None, 'gradient_clip': None, 'amp': False, 'input_standardization': 'training_partition_mean_population_std', 'rng': 'epoch_dropout_seed_and_separate_local_loader_generator_v1', 'cudnn_deterministic': True, 'cudnn_benchmark': False, 'matmul_tf32': False, 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__), 'numpy': str(np.__version__), 'sklearn': str(sklearn.__version__), 'device_type': device.type, 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu', 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
    path = out / 'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != manifest:
            keys = [k for k in set(previous) | set(manifest) if previous.get(k) != manifest.get(k)]
            raise RuntimeError(f'Saved experiment settings differ ({keys}). Restore them or use a NEW output folder.')
    else:
        atomic_json(path, manifest)

def run_experiment():
    configure_torch()
    device = torch.device(DEVICE)
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    if len(PANEL) != 5 or len(set(PANEL)) != 5 or any((a not in ACTIVATIONS for a in PANEL)):
        raise RuntimeError('The new panel must contain five registered activations exactly once.')
    data = FashionData(DATA_DIR, out)
    ensure_manifest(out, data, device)
    _EPOCH_CACHE.clear()
    done = merge_outputs(out)
    print(f'Fashion-MNIST BN depth {DEPTH}: {len(PANEL)} functions x {len(SEEDS)} seeds = {len(PANEL) * len(SEEDS)} runs; {done} already finalized. Output: {out.resolve()}', flush=True)
    print('Logish seed 42 is first. Interrupt normally to pause; rerun this cell to resume.', flush=True)
    try:
        for seed in SEEDS:
            for name in PANEL:
                run_one(name, seed, data, out, device)
    except KeyboardInterrupt:
        print('\nPaused. The unfinished epoch will replay from the last saved epoch on resume.', flush=True)
    finally:
        merge_outputs(out)
    print(f'Saved epochs.csv, results.csv, summary.csv and run_status.csv in {out.resolve()}', flush=True)
run_experiment()
