"""MNIST BN35 diagnostic reruns; Mish, Swish, BandHigh, BandLow; seeds 42 and 43.

Outputs are isolated under runs/diagnostic_mnist_bn35.
Completed-run resume logic and scientific training settings are retained.
"""

import os
import random
import time
from functools import partial
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
from layerwise_diagnostics import check_against_table1, layerwise_snapshot, make_fixed_batches
BASE_DIR = os.environ.get('ACTIVATION_OUTPUT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'diagnostic_mnist_bn35'))
os.makedirs(BASE_DIR, exist_ok=True)
DATA_DIR = os.environ.get('ACTIVATION_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'))
RESULTS_PATH = os.path.join(BASE_DIR, 'diag_results.csv')
EPOCHS_PATH = os.path.join(BASE_DIR, 'diag_epochs.csv')
SUMMARY_PATH = os.path.join(BASE_DIR, 'diag_summary.csv')
LAYERWISE_PATH = os.path.join(BASE_DIR, 'diag_layerwise.csv')
os.makedirs(DATA_DIR, exist_ok=True)
N_LAYERS = 35
HIDDEN_DIM = 500
DROPOUT = 0.25
SEEDS = [42, 43]
SPLIT_SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_WORKERS = int(os.environ.get('ACTIVATION_NUM_WORKERS', '0'))
MIN_EPOCHS = 50
MAX_EPOCHS = 100
PATIENCE = 20
NORM_MODE = 'batchnorm'
DIAG_EPOCHS = tuple((int(q * MAX_EPOCHS) for q in (0.1, 0.2, 0.5)))
DIAG_N_BATCHES = 4
DIAG_SEED = 0

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
split = StratifiedShuffleSplit(n_splits=1, test_size=1 / 6, random_state=SPLIT_SEED)
train_idx, val_idx = next(split.split(np.zeros(len(train_targets)), train_targets))
train_loader = DataLoader(Subset(full_train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(Subset(full_train_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f'Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_dataset)}')
diag_batches = make_fixed_batches(full_train_dataset, train_idx, BATCH_SIZE, DIAG_N_BATCHES, device)

class MishAct(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SwishAct(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)

class GatedTwoBumps(nn.Module):

    def __init__(self, beta, b, alpha1, alpha2):
        super().__init__()
        self.beta = beta
        self.b = b
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha1 * wide + self.alpha2 * narrow
ACTIVATIONS = {'Mish': MishAct, 'Swish': SwishAct, 'BandHigh': partial(GatedTwoBumps, beta=1.20680028, b=0.04461946, alpha1=0.16568067, alpha2=-0.07683369), 'BandLow': partial(GatedTwoBumps, beta=3.09266783, b=4.56834656, alpha1=-0.47355169, alpha2=0.08382025)}
check_against_table1(ACTIVATIONS)

class DeepMLP(nn.Module):

    def __init__(self, make_activation, n_layers, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.activation = make_activation()
        in_dim = 784
        linears, norms = ([], [])
        for _ in range(n_layers):
            lin = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            linears.append(lin)
            norms.append(nn.BatchNorm1d(hidden_dim, affine=True))
            in_dim = hidden_dim
        self.linears = nn.ModuleList(linears)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(DROPOUT)
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

def mean_abs_gamma(model):
    vals = [norm.weight.detach().abs().mean().item() for norm in model.norms]
    return round(float(np.mean(vals)), 4)

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
            loss = criterion(model(inputs), labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    return total_loss / total

def train_one_run(act_name, make_activation, n_layers, seed):
    set_all_seeds(seed)
    model = DeepMLP(make_activation, n_layers).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    stopped_at = MAX_EPOCHS
    first_improvement_epoch = None
    baseline_val_loss = None
    best_path = f'_best_{act_name}_{NORM_MODE}_d{n_layers}_s{seed}.pt'
    epoch_rows = []
    diag_rows = []

    def snapshot(stage):
        t0 = time.time()
        for r in layerwise_snapshot(model, diag_batches, make_activation, DIAG_SEED):
            diag_rows.append({'activation': act_name, 'norm_mode': NORM_MODE, 'n_layers': n_layers, 'seed': seed, 'stage': stage, **r})
        print(f"   [diag] snapshot '{stage}' done in {time.time() - t0:.1f}s", flush=True)
    print(f'\n>>> {act_name} | depth={n_layers} | seed={seed} | patience={PATIENCE}')
    start = time.time()
    snapshot('init')
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = (inputs.to(device), labels.to(device))
            optimizer.zero_grad()
            criterion(model(inputs), labels).backward()
            optimizer.step()
        val_loss = evaluate_loss(model, val_loader, criterion)
        val_acc = evaluate_acc(model, val_loader)
        epoch_rows.append({'activation': act_name, 'norm_mode': NORM_MODE, 'n_layers': n_layers, 'seed': seed, 'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc})
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
            print(f'   Epoch {epoch:02d}/{MAX_EPOCHS} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | patience {epochs_no_improve}/{PATIENCE} | mean|gamma|={mean_abs_gamma(model)}')
        if epoch in DIAG_EPOCHS:
            snapshot(f'epoch{epoch}')
        if epochs_no_improve >= PATIENCE and epoch >= MIN_EPOCHS:
            stopped_at = epoch
            print(f'   >>> Early stop at epoch {epoch}')
            break
    model.load_state_dict(torch.load(best_path))
    os.remove(best_path)
    test_acc = evaluate_acc(model, test_loader)
    gamma_final = mean_abs_gamma(model)
    snapshot('best')
    val_acc_at_epoch_50 = next((r['val_acc'] for r in epoch_rows if r['epoch'] == 50), float('nan'))
    print(f'   FINAL test_acc={test_acc:.4f} | val_acc@50={val_acc_at_epoch_50:.4f} | best_val_loss={best_val_loss:.4f} | stopped_at={stopped_at} | mean|gamma|_final={gamma_final} | time={time.time() - start:.1f}s')
    return (test_acc, gamma_final, stopped_at, first_improvement_epoch, val_acc_at_epoch_50, epoch_rows, diag_rows)
RUN_SPECS = [(act, NORM_MODE, N_LAYERS, seed) for act in ACTIVATIONS for seed in SEEDS]
print(f'Total planned runs: {len(RUN_SPECS)}')
all_results, all_epochs = ([], [])
if os.path.exists(RESULTS_PATH):
    try:
        all_results = pd.read_csv(RESULTS_PATH).to_dict('records')
        print(f'Loaded {len(all_results)} existing results from CSV.')
    except Exception as e:
        print(f'Could not load results CSV ({e}). Starting fresh.')
if os.path.exists(EPOCHS_PATH):
    try:
        all_epochs = pd.read_csv(EPOCHS_PATH).to_dict('records')
        print(f'Loaded {len(all_epochs)} existing epoch rows from CSV.')
    except Exception as e:
        print(f'Could not load epochs CSV ({e}). Starting fresh.')
completed = {(str(r['activation']), str(r['norm_mode']), int(r['n_layers']), int(r['seed'])) for r in all_results}
all_diag = []
if os.path.exists(LAYERWISE_PATH):
    try:
        all_diag = pd.read_csv(LAYERWISE_PATH).to_dict('records')
        all_diag = [r for r in all_diag if (str(r['activation']), str(r['norm_mode']), int(r['n_layers']), int(r['seed'])) in completed]
        print(f'Loaded {len(all_diag)} existing layerwise rows from CSV.')
    except Exception as e:
        print(f'Could not load layerwise CSV ({e}). Starting fresh.')
total = len(RUN_SPECS)
done = len(completed)
print(f'Already done: {done}/{total}\n')
for act_name, norm_mode, n_layers, seed in RUN_SPECS:
    if (act_name, norm_mode, n_layers, seed) in completed:
        print(f'Skipping {act_name} | depth={n_layers} | seed={seed}')
        continue
    test_acc, gamma_final, stopped_at, first_improvement_epoch, val_acc_at_epoch_50, epoch_rows, diag_rows = train_one_run(act_name, ACTIVATIONS[act_name], n_layers, seed)
    done += 1
    all_diag.extend(diag_rows)
    pd.DataFrame(all_diag).to_csv(LAYERWISE_PATH, index=False)
    all_results.append({'activation': act_name, 'norm_mode': norm_mode, 'n_layers': n_layers, 'seed': seed, 'test_acc': round(test_acc, 4), 'val_acc_at_epoch_50': round(val_acc_at_epoch_50, 4) if np.isfinite(val_acc_at_epoch_50) else float('nan'), 'stopped_at': stopped_at, 'first_improvement_epoch': first_improvement_epoch, 'mean_abs_gamma_final': gamma_final, 'patience_used': PATIENCE})
    pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)
    all_epochs.extend(epoch_rows)
    pd.DataFrame(all_epochs).to_csv(EPOCHS_PATH, index=False)
    print(f'[{done}/{total}] Saved.')
print('\n' + '=' * 60)
print('ALL DONE')
print('=' * 60)
df = pd.read_csv(RESULTS_PATH)
summary = df.groupby(['activation', 'norm_mode', 'n_layers'])['test_acc'].mean().round(4)
print('\nMean test accuracy by activation:')
print(summary.to_string())
summary.reset_index().to_csv(SUMMARY_PATH, index=False)
print(f'Saved {os.path.basename(SUMMARY_PATH)}')
