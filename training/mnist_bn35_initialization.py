"""MNIST BN35; paired Xavier-normal and variance-matched orthogonal initialization.
Eight activations, two initializers, five seeds: 80 performance runs.

Five seeds per activation; source training and resume rules retained.
"""

from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import csv, hashlib, inspect, json, math, platform, random, sys, time, types
import importlib.metadata
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
OUTPUT_DIR = __import__('pathlib').Path(__file__).resolve().parents[1] / 'runs/mnist_bn35_initialization'
DATA_DIR = __import__('pathlib').Path(__file__).resolve().parents[1] / 'datasets'
RUN_MODE = 'all'
HALT_ON_NONFINITE = True
MAX_TOTAL_HOURS = 200.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0
SEEDS = [42, 43, 44, 45, 46]
ACTIVATION_PANEL = ['Mish', 'Swish', 'ScaledSwish', 'ScaledSwishhalf', 'SetD5_HighA_LowGamma', 'SetD6_HighA_HighGamma', 'SetD1_LowA_HighGamma', 'MatchAllToMish']
INITIALIZERS = ['xavier_normal', 'orthogonal_matched']
DEPTH, WIDTH, DROPOUT = (35, 500, 0.25)
BATCH_SIZE, LEARNING_RATE = (128, 0.01)
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = (100, 50, 20)
SCREEN_STEPS = 20
VERSION = 'BN35_init_methods_1.0'
PARAMETERS = {'MatchAllToMish': (2.0996330505, 2.5444438979, -0.3271993629, 0.0), 'SetD1_LowA_HighGamma': (0.4949478828, -0.413756283, 0.0388663503, 0.0027885451), 'SetD5_HighA_LowGamma': (0.25, 1.6646126199, -0.2443422477, -0.015827546), 'SetD6_HighA_HighGamma': (0.3558473158, 1.1158205221, -0.1377719114, -0.009294866)}
LITERATURE = ['Mish', 'Swish']
ALL_ACTIVATIONS = ['Mish', 'Swish', 'ScaledSwish', 'ScaledSwishhalf'] + list(PARAMETERS)
_BUDGET = None
_BANK_CACHE = {}

class Activation(nn.Module):

    def __init__(self, name):
        super().__init__()
        if name not in ALL_ACTIVATIONS:
            raise ValueError(name)
        self.name = name

    def forward(self, x):
        if self.name == 'Mish':
            return x * torch.tanh(F.softplus(x))
        if self.name in ('Swish', 'ScaledSwish', 'ScaledSwishhalf'):
            scale = {'Swish': 1.0, 'ScaledSwish': 1.4, 'ScaledSwishhalf': 0.5}[self.name]
            return scale * x * torch.sigmoid(x)
        beta, b, alpha, eta = PARAMETERS[self.name]
        wide = x * torch.exp(-0.5 * x ** 2)
        y = x * torch.sigmoid(beta * x + b) + alpha * wide
        if eta:
            narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
            y = y + eta * narrow
        return y

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class NumericalFailure(RuntimeError):
    pass

def check_tensor(tensor, where):
    if not torch.isfinite(tensor).all().item():
        raise NumericalFailure(f'Non-finite {where}')

def check_named_tensors(named, where):
    values = [(n, t) for n, t in named if t is not None and t.is_floating_point()]
    if not values:
        return
    flags = torch.stack([torch.isfinite(t).all() for _, t in values])
    if not flags.all().item():
        bad = [n for (n, _), ok in zip(values, flags.cpu().tolist()) if not ok]
        raise NumericalFailure(f"Non-finite {where}: {', '.join(bad)}")

@contextmanager
def forward_guards(model):
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.BatchNorm1d, Activation)):

            def hook(_module, _inputs, output, name=name):
                check_tensor(output, f'forward output at {name}')
            handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temp, path)

def atomic_checkpoint(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def state_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def worker_seed(_worker_id):
    value = torch.initial_seed() % 2 ** 32
    random.seed(value)
    np.random.seed(value)

class Data:

    def __init__(self, args, out):
        from sklearn.model_selection import StratifiedShuffleSplit
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full = datasets.MNIST(args.data, train=True, download=True, transform=transform)
        self.test_set = datasets.MNIST(args.data, train=False, download=True, transform=transform)
        targets = np.asarray(full.targets)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 / 6, random_state=42)
        train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
        split_path = out / 'split_indices.npz'
        if split_path.exists():
            with np.load(split_path) as saved:
                if not (np.array_equal(train_idx, saved['train']) and np.array_equal(val_idx, saved['validation'])):
                    raise RuntimeError('Dataset split changed; use a new output directory.')
        else:
            np.savez(split_path, train=train_idx, validation=val_idx)
        self.train_set, self.val_set = (Subset(full, train_idx), Subset(full, val_idx))
        self.split_hash = hashlib.sha256(train_idx.tobytes() + val_idx.tobytes()).hexdigest()
        g = torch.Generator().manual_seed(0)
        self.diag = next(iter(DataLoader(self.train_set, batch_size=128, shuffle=True, generator=g, num_workers=0)))
        self.args = args
        print(f'MNIST: train={len(train_idx)} validation={len(val_idx)} test={len(self.test_set)}', flush=True)

    def loader(self, part, seed, epoch=0):
        dataset = {'train': self.train_set, 'validation': self.val_set, 'test': self.test_set}[part]
        g = torch.Generator().manual_seed(seed + 100003 * epoch + 17)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=part == 'train', num_workers=self.args.workers, generator=g, worker_init_fn=worker_seed, pin_memory=self.args.device.startswith('cuda'))

def train_epoch(model, optimizer, loader, device, max_steps=None):
    model.train()
    total_loss, correct, count, maximum_gradient = (0.0, 0, 0, 0.0)
    for step, (x, y) in enumerate(loader, 1):
        budget_check()
        try:
            x, y = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            check_tensor(logits, 'training logits')
            loss = F.cross_entropy(logits, y)
            check_tensor(loss, 'training loss')
            loss.backward()
            gradients = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
            check_named_tensors(gradients, 'gradients before optimizer.step')
            grad_max = torch.stack([g.detach().abs().max() for _, g in gradients]).max().item()
            maximum_gradient = max(maximum_gradient, grad_max)
            optimizer.step()
            check_named_tensors(model.named_parameters(), 'parameters after optimizer.step')
            check_named_tensors(model.named_buffers(), 'model buffers')
        except NumericalFailure as exc:
            raise NumericalFailure(f'training batch {step}: {exc}') from exc
        total_loss += float(loss.item()) * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
        if max_steps is not None and step >= max_steps:
            break
    return {'train_loss': total_loss / count, 'train_acc': correct / count, 'max_abs_gradient': maximum_gradient, 'train_examples': count}

@torch.no_grad()
def evaluate(model, loader, device, max_steps=None):
    model.eval()
    loss_total, correct, count = (0.0, 0, 0)
    for step, (x, y) in enumerate(loader, 1):
        budget_check()
        x, y = (x.to(device), y.to(device))
        logits = model(x)
        check_tensor(logits, f'evaluation logits, batch {step}')
        loss = F.cross_entropy(logits, y)
        check_tensor(loss, f'evaluation loss, batch {step}')
        loss_total += loss.item() * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
        if max_steps is not None and step >= max_steps:
            break
    return {'loss': loss_total / count, 'acc': correct / count, 'examples': count}

def diagnostics(model, batch, device, mode):
    flags = [(m, m.training) for m in model.modules()]
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    rows = []
    try:
        model.eval()
        if mode == 'batch_stats':
            for norm in model.norms:
                if isinstance(norm, nn.BatchNorm1d):
                    norm.train()
        selected = {0, len(model.linears) // 2, len(model.linears) - 1}
        h = batch[0].to(device).reshape(len(batch[0]), -1)
        for i, (linear, norm) in enumerate(zip(model.linears, model.norms)):
            with torch.no_grad():
                pre = linear(h)
                z = norm(pre)
                h = model.activation(z)
            check_tensor(h, f'diagnostic hidden layer {i + 1}')
            if i in selected:
                zg = z.detach().requires_grad_(True)
                derivative = torch.autograd.grad(model.activation(zg).sum(), zg)[0]
                check_tensor(derivative, f'diagnostic derivative layer {i + 1}')
                w = linear.weight.detach()
                row = {'layer': i + 1, 'bn_mode': mode, 'pre_bn_std': pre.std(unbiased=False).item(), 'activation_input_mean': z.mean().item(), 'activation_input_std': z.std(unbiased=False).item(), 'activation_output_std': h.std(unbiased=False).item(), 'mean_derivative': derivative.mean().item(), 'mean_squared_derivative': derivative.double().square().mean().item(), 'weight_frobenius': w.double().norm().item()}
                if isinstance(norm, nn.BatchNorm1d):
                    row.update(bn_gamma_absmean=norm.weight.detach().abs().mean().item(), bn_running_var_mean=norm.running_var.mean().item())
                rows.append(row)
        with torch.no_grad():
            logits = model.output(h)
        check_tensor(logits, 'diagnostic logits')
        rows.append({'layer': 'output', 'bn_mode': mode, 'logit_std': logits.std(unbiased=False).item(), 'logit_absmax': logits.abs().max().item()})
    finally:
        with torch.no_grad():
            for name, buf in model.named_buffers():
                buf.copy_(buffers[name])
        for module, flag in flags:
            module.training = flag
    return rows

def optional_diagnostics(model, data, args, stage):
    rows = []
    for mode in ['batch_stats', 'running_stats'] if args.norm == 'batchnorm' else ['running_stats']:
        try:
            rows.extend((dict(row, stage=stage, diagnostic_status='ok') for row in diagnostics(model, data.diag, args.device, mode)))
        except NumericalFailure as exc:
            rows.append({'stage': stage, 'bn_mode': mode, 'diagnostic_status': 'nonfinite', 'message': str(exc)})
    return rows

def activation_risk(names):
    rows = []
    for name in names:
        x = torch.linspace(-12, 12, 24001, dtype=torch.float64, requires_grad=True)
        y = Activation(name)(x)
        d = torch.autograd.grad(y.sum(), x)[0]
        tail = torch.tensor([-1000.0, -100.0, -20.0, 0.0, 20.0, 100.0, 1000.0], requires_grad=True)
        ty = Activation(name)(tail)
        td = torch.autograd.grad(ty.sum(), tail)[0]
        finite = bool(torch.isfinite(y).all() and torch.isfinite(d).all() and torch.isfinite(ty).all() and torch.isfinite(td).all())
        rows.append({'activation': name, 'formula_finite': finite, 'max_abs_derivative_on_grid': float(d.abs().max()) if finite else None, 'max_abs_output_on_grid': float(y.abs().max()) if finite else None, 'nested_exponential': name in ('TanhExpAct', 'ExpExpishAct')})
    rows.sort(key=lambda r: (not r['formula_finite'], r['nested_exponential'], r['max_abs_derivative_on_grid'] or 0), reverse=True)
    return rows

class DeepMLP(nn.Module):

    def __init__(self, activation, depth=35, width=500, norm='batchnorm', dropout=0.25):
        super().__init__()
        self.activation = Activation(activation)
        self.linears, self.norms = (nn.ModuleList(), nn.ModuleList())
        incoming = 784
        for _ in range(depth):
            linear = nn.Linear(incoming, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)
            self.norms.append(nn.BatchNorm1d(width, eps=1e-05, momentum=0.1) if norm == 'batchnorm' else nn.Identity())
            incoming = width
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width, 10)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for linear, norm in zip(self.linears, self.norms):
            x = self.dropout(self.activation(norm(linear(x))))
        return self.output(x)

def condition_id(activation, initializer):
    return f'{activation}__{initializer}'

def run_id(activation, initializer, seed):
    return f'{condition_id(activation, initializer)}__s{seed}'

def screen_one(args, data, activation, initializer, seed, out):
    budget_check()
    path = out / 'screen' / f'{condition_id(activation, initializer)}.json'
    if path.exists():
        return read_json(path)
    model = build_model(args, activation, seed, initializer)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    row = {'activation': activation, 'initializer': initializer, 'seed': seed, 'screen_steps': args.screen_steps, 'status': 'ok'}
    started = time.time()
    try:
        seed_all(seed + 1000003)
        with forward_guards(model):
            row.update(train_epoch(model, optimizer, data.loader('train', seed, 1), args.device, max_steps=args.screen_steps))
            val = evaluate(model, data.loader('validation', seed), args.device, max_steps=4)
        row.update(screen_val_loss=val['loss'], screen_val_acc=val['acc'])
    except NumericalFailure as exc:
        row.update(status='nonfinite', message=str(exc))
    row['seconds'] = time.time() - started
    atomic_json(path, row)
    print(f"SCREEN {activation:28} {initializer}: {row['status']} grad_max={row.get('max_abs_gradient', 'n/a')} {row.get('message', '')}", flush=True)
    return row

def train_run(args, data, activation, initializer, seed, out):
    budget_check()
    rid = run_id(activation, initializer, seed)
    folder = out / 'runs' / rid
    folder.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint_path = (folder / 'result.json', folder / 'last.pt')
    if result_path.exists():
        return read_json(result_path)
    failure_path = folder / 'failure.json'
    if failure_path.exists():
        return read_json(failure_path)
    base = {'run_id': rid, 'activation': activation, 'group': 'literature' if activation in LITERATURE else 'constructed', 'initializer': initializer, 'seed': seed, 'norm_mode': args.norm, 'n_layers': args.depth, 'width': args.width}
    model = build_model(args, activation, seed, initializer)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    history, diag_rows, best_state = ([], [], None)
    best_loss, best_epoch, stale, last_epoch = (math.inf, None, 0, 0)
    train_seconds, training_finished = (0.0, False)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history, diag_rows = (checkpoint['history'], checkpoint['diagnostics'])
        best_state, best_loss = (checkpoint['best_model'], checkpoint['best_loss'])
        best_epoch, stale = (checkpoint['best_epoch'], checkpoint['stale'])
        last_epoch, training_finished = (checkpoint['epoch'], checkpoint['training_finished'])
        train_seconds = checkpoint['training_seconds']
        del checkpoint
        print(f'RESUME {rid}: after epoch {last_epoch}', flush=True)
    else:
        diag_rows = optional_diagnostics(model, data, args, 'initialization')
    atomic_csv(folder / 'epochs.csv', [dict(base, **row) for row in history])
    atomic_csv(folder / 'diagnostics.csv', [dict(base, **row) for row in diag_rows])
    stage, epoch = ('training', last_epoch)
    started = time.time()
    try:
        if not training_finished:
            for epoch in range(last_epoch + 1, args.epochs + 1):
                budget_check()
                epoch_started = time.time()
                seed_all(seed + 1000003 * epoch)
                train = train_epoch(model, optimizer, data.loader('train', seed, epoch), args.device)
                stage = 'validation'
                val = evaluate(model, data.loader('validation', seed), args.device)
                if val['loss'] < best_loss:
                    best_loss, best_epoch, stale = (val['loss'], epoch, 0)
                    best_state = state_copy(model)
                else:
                    stale += 1
                if epoch == args.min_epochs:
                    stale = 0
                history.append({'epoch': epoch, **train, 'val_loss': val['loss'], 'val_acc': val['acc'], 'best_epoch_so_far': best_epoch, 'epoch_train_val_seconds': time.time() - epoch_started})
                if epoch in {1, math.ceil(args.epochs / 10), math.ceil(args.epochs / 5), math.ceil(args.epochs / 2), args.epochs}:
                    diag_rows.extend(optional_diagnostics(model, data, args, f'epoch_{epoch}'))
                training_finished = epoch == args.epochs or (epoch >= args.min_epochs and stale >= args.patience)
                checkpoint = {'model': state_copy(model), 'optimizer': optimizer.state_dict(), 'best_model': best_state, 'history': history, 'diagnostics': diag_rows, 'best_loss': best_loss, 'best_epoch': best_epoch, 'stale': stale, 'epoch': epoch, 'training_finished': training_finished, 'training_seconds': train_seconds + time.time() - started}
                atomic_checkpoint(checkpoint_path, checkpoint)
                budget_flush()
                del checkpoint
                atomic_csv(folder / 'epochs.csv', [dict(base, **row) for row in history])
                atomic_csv(folder / 'diagnostics.csv', [dict(base, **row) for row in diag_rows])
                print(f"{rid} epoch {epoch:3}/{args.epochs}: train={train['train_loss']:.5f} val={val['loss']:.5f} acc={val['acc']:.4f} max|grad|={train['max_abs_gradient']:.3g}", flush=True)
                if training_finished:
                    break
                stage = 'training'
        stage = 'final_checkpoint_test'
        final_test = evaluate(model, data.loader('test', seed), args.device)
        model.load_state_dict(best_state)
        stage = 'validation_selected_checkpoint_test'
        best_test = evaluate(model, data.loader('test', seed), args.device)
        diag_rows.extend(optional_diagnostics(model, data, args, 'validation_selected'))
        result = {**base, 'status': 'completed', 'stopped_at': epoch, 'stop_reason': 'budget' if epoch == args.epochs else 'patience', 'best_epoch': best_epoch, 'best_val_loss': best_loss, 'best_val_acc': history[best_epoch - 1]['val_acc'], 'test_loss': best_test['loss'], 'test_acc': best_test['acc'], 'final_val_loss': history[-1]['val_loss'], 'final_val_acc': history[-1]['val_acc'], 'final_test_loss': final_test['loss'], 'final_test_acc': final_test['acc'], 'training_and_finish_seconds': train_seconds + time.time() - started}
        for percent, denominator in [(10, 10), (20, 5), (50, 2)]:
            target = math.ceil(args.epochs / denominator)
            row = next((r for r in history if r['epoch'] == target), None)
            result[f'val_loss_at_{percent}pct'] = row['val_loss'] if row else None
            result[f'val_acc_at_{percent}pct'] = row['val_acc'] if row else None
        atomic_csv(folder / 'diagnostics.csv', [dict(base, **row) for row in diag_rows])
        atomic_checkpoint(folder / 'best.pt', {'model': best_state, 'run': base, 'epoch': best_epoch})
        atomic_json(result_path, result)
        budget_flush()
        checkpoint_path.unlink(missing_ok=True)
        print(f"DONE {rid}: test_acc={result['test_acc']:.4f}, best_epoch={best_epoch}", flush=True)
        return result
    except NumericalFailure as exc:
        failure = {**base, 'status': 'nonfinite', 'failure_epoch': epoch, 'failure_stage': stage, 'message': str(exc), 'last_completed_epoch': history[-1]['epoch'] if history else 0, 'best_val_loss_before_failure': best_loss if math.isfinite(best_loss) else None}
        atomic_json(failure_path, failure)
        budget_flush()
        print(f'FAILED {rid}: {exc}', flush=True)
        return failure

class BudgetStop(RuntimeError):
    pass

class ActiveBudget:

    def __init__(self, path, max_hours):
        self.path = Path(path)
        self.previous = read_json(path)['active_seconds'] if self.path.exists() else 0.0
        self.started = time.monotonic()
        self.last_flush = self.started
        self.limit = None if max_hours is None else max_hours * 3600

    def elapsed(self):
        return self.previous + time.monotonic() - self.started

    def flush(self):
        atomic_json(self.path, {'active_seconds': self.elapsed(), 'limit_hours': None if self.limit is None else self.limit / 3600})
        self.last_flush = time.monotonic()

    def check(self):
        if self.limit is not None and self.elapsed() >= self.limit:
            self.flush()
            raise BudgetStop('Cumulative active-time budget reached; no condition was shortened or marked complete.')
        if time.monotonic() - self.last_flush >= 60:
            self.flush()

def budget_check():
    if _BUDGET is not None:
        _BUDGET.check()

def budget_flush():
    if _BUDGET is not None:
        _BUDGET.flush()

@contextmanager
def directory_lock(out):
    stream = (out / 'experiment.lock').open('a+b')
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another process is using OUTPUT_DIR. Do not run two copies there.') from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Another process is using OUTPUT_DIR. Do not run two copies there.') from exc
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == 'nt':
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

def orthogonalize(weight):
    fan_out, fan_in = weight.shape
    transpose = fan_out < fan_in
    matrix = weight.detach().cpu().double()
    q, r = torch.linalg.qr(matrix.T if transpose else matrix, mode='reduced')
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(0)
    if transpose:
        q = q.T
    gain = math.sqrt(2 * max(fan_in, fan_out) / (fan_in + fan_out))
    return ((q * gain).to(weight.dtype).contiguous(), gain)

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def initialization_bank(args, seed):
    key = (str(args.out), seed, args.depth, args.width, args.dropout)
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    folder = Path(args.out) / 'initializations'
    folder.mkdir(parents=True, exist_ok=True)
    paths = {method: folder / f'seed{seed}_{method}.pt' for method in INITIALIZERS}
    meta_path = folder / f'seed{seed}_hashes.json'
    if meta_path.exists():
        hashes = read_json(meta_path)
        for method, path in paths.items():
            if not path.exists() or file_sha256(path) != hashes[method]:
                raise RuntimeError(f'Initialization file missing/changed: {path}; do not mix results.')
        bank = {method: torch.load(path, map_location='cpu', weights_only=True) for method, path in paths.items()}
    else:
        budget_check()
        seed_all(seed)
        original = DeepMLP('Mish', args.depth, args.width, args.norm, args.dropout)
        xavier = state_copy(original)
        orthogonal = {k: v.clone() for k, v in xavier.items()}
        audit = []
        for i, linear in enumerate(original.linears):
            budget_check()
            name = f'linears.{i}.weight'
            orthogonal[name], gain = orthogonalize(xavier[name])
            fan_out, fan_in = linear.weight.shape
            w = orthogonal[name].double()
            gram = w @ w.T if fan_out < fan_in else w.T @ w
            target = torch.eye(min(fan_in, fan_out), dtype=torch.float64) * gain ** 2
            error = (gram - target).abs().max().item()
            if error > 1e-05:
                raise RuntimeError(f'Orthogonal construction failed its Gram check: {name}, {error}')
            audit.append({'seed': seed, 'layer': i + 1, 'fan_in': fan_in, 'fan_out': fan_out, 'orthogonal_gain': gain, 'target_entry_second_moment': 2 / (fan_in + fan_out), 'xavier_actual_entry_second_moment': xavier[name].double().square().mean().item(), 'orthogonal_actual_entry_second_moment': w.square().mean().item(), 'orthogonal_gram_max_error': error, 'xavier_weight_hash': hashlib.sha256(xavier[name].numpy().tobytes()).hexdigest(), 'orthogonal_weight_hash': hashlib.sha256(orthogonal[name].numpy().tobytes()).hexdigest()})
        for name in xavier:
            if not (name.startswith('linears.') and name.endswith('.weight')):
                if not torch.equal(xavier[name], orthogonal[name]):
                    raise AssertionError(f'Unintended initializer difference: {name}')
        bank = {'xavier_normal': xavier, 'orthogonal_matched': orthogonal}
        for method, path in paths.items():
            atomic_checkpoint(path, bank[method])
        atomic_csv(folder / f'seed{seed}_audit.csv', audit)
        atomic_json(meta_path, {method: file_sha256(path) for method, path in paths.items()})
        print(f'Initialization bank ready, seed {seed}; variance match and orthogonality checked.', flush=True)
    _BANK_CACHE.clear()
    _BANK_CACHE[key] = bank
    return bank

def build_model(args, activation, seed, initializer):
    bank = initialization_bank(args, seed)
    seed_all(seed)
    model = DeepMLP(activation, args.depth, args.width, args.norm, args.dropout)
    model.load_state_dict(bank[initializer])
    return model.to(args.device)

def normalized_code(code):

    def constant(value):
        if isinstance(value, types.CodeType):
            return normalized_code(value)
        if isinstance(value, tuple):
            return [constant(v) for v in value]
        if isinstance(value, frozenset):
            return sorted([constant(v) for v in value], key=repr)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, bytes):
            return {'bytes': value.hex()}
        return repr(value)
    return {'bytecode': code.co_code.hex(), 'constants': [constant(c) for c in code.co_consts], 'names': code.co_names, 'varnames': code.co_varnames, 'freevars': code.co_freevars, 'cellvars': code.co_cellvars, 'argcount': code.co_argcount, 'kwonly': code.co_kwonlyargcount, 'posonly': code.co_posonlyargcount, 'flags': code.co_flags, 'exceptiontable': getattr(code, 'co_exceptiontable', b'').hex()}

def implementation_hash():
    bodies = {}
    for name in IMPLEMENTATION_OBJECTS:
        obj = globals()[name]
        if inspect.isclass(obj):
            bodies[name] = {k: normalized_code(inspect.unwrap(v).__code__) for k, v in vars(obj).items() if inspect.isfunction(v)}
        else:
            bodies[name] = normalized_code(inspect.unwrap(obj).__code__)
    return hashlib.sha256(json.dumps(bodies, sort_keys=True).encode()).hexdigest()

def make_args():
    return SimpleNamespace(out=Path(OUTPUT_DIR).resolve(), data=Path(DATA_DIR).resolve(), device=DEVICE, workers=NUM_WORKERS, depth=DEPTH, width=WIDTH, dropout=DROPOUT, norm='batchnorm', batch_size=BATCH_SIZE, lr=LEARNING_RATE, epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS, patience=PATIENCE, screen_steps=SCREEN_STEPS)

def ensure_manifest(args):
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'scikit-learn'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'out'}
    config.update(version=VERSION, activations=ACTIVATION_PANEL, seeds=SEEDS, initializers=INITIALIZERS, activation_parameters=PARAMETERS, initialization_scope='hidden_linear_weights_only', variance_matching='expected_entry_second_moment_2_over_fanin_plus_fanout', main_fixed_budget_outcome='validation_loss_at_epoch_50', split_seed=42, optimizer='SGD_no_momentum_no_weight_decay', bn_eps=1e-05, bn_momentum=0.1, bn_affine=True, implementation_hash=implementation_hash(), packages=packages, python=f'{sys.version_info.major}.{sys.version_info.minor}', cuda_version=torch.version.cuda, cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(), device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else platform.machine())
    config = json.loads(json.dumps(config))
    path = args.out / 'manifest.json'
    if path.exists():
        previous = read_json(path)['config']
        if previous != config:
            changed = [k for k in set(previous) | set(config) if previous.get(k) != config.get(k)]
            raise RuntimeError(f'Protocol/code/environment changed: {changed}. Use a NEW OUTPUT_DIR; do not append incompatible runs to this comparison.')
    else:
        atomic_json(path, {'config': config, 'planned_formal_runs': len(ACTIVATION_PANEL) * len(INITIALIZERS) * len(SEEDS), 'note': 'Paired initialization experiment; results are generated independently for both initializers.'})

def aggregate(out):
    results, epochs, diag_rows = ([], [], [])
    for folder in sorted((out / 'runs').glob('*')):
        for filename in ('result.json', 'failure.json'):
            if (folder / filename).exists():
                results.append(read_json(folder / filename))
                break
        for filename, target in [('epochs.csv', epochs), ('diagnostics.csv', diag_rows)]:
            if (folder / filename).exists():
                with (folder / filename).open(newline='', encoding='utf-8') as stream:
                    target.extend(csv.DictReader(stream))
    atomic_csv(out / 'results.csv', results)
    atomic_csv(out / 'epochs.csv', epochs)
    atomic_csv(out / 'diagnostics.csv', diag_rows)
    indexed = {(r['activation'], r['initializer'], r['seed']): r for r in results}
    status_rows, summary, paired = ([], [], [])
    metrics = ['val_loss_at_10pct', 'val_loss_at_20pct', 'val_loss_at_50pct', 'val_acc_at_50pct', 'best_val_loss', 'test_loss', 'test_acc', 'final_val_loss', 'final_test_loss', 'final_test_acc']
    for name in ACTIVATION_PANEL:
        for method in INITIALIZERS:
            group = []
            for seed in SEEDS:
                rid = run_id(name, method, seed)
                result = indexed.get((name, method, seed))
                state = result['status'] if result else 'incomplete_checkpoint' if (out / 'runs' / rid / 'last.pt').exists() else 'pending'
                status_rows.append({'run_id': rid, 'activation': name, 'initializer': method, 'seed': seed, 'status': state})
                if result:
                    group.append(result)
            good = [r for r in group if r['status'] == 'completed']
            row = {'activation': name, 'initializer': method, 'expected_seeds': len(SEEDS), 'completed': len(good), 'numerical_failures': len(group) - len(good), 'pending_or_incomplete': len(SEEDS) - len(group)}
            for metric in metrics:
                values = [r[metric] for r in good if r.get(metric) is not None]
                row[metric + '_mean_successful_runs'] = float(np.mean(values)) if values else None
                row[metric + '_sd_successful_runs'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(row)
        for seed in SEEDS:
            a = indexed.get((name, 'xavier_normal', seed))
            b = indexed.get((name, 'orthogonal_matched', seed))
            row = {'activation': name, 'seed': seed, 'xavier_status': a['status'] if a else 'pending', 'orthogonal_status': b['status'] if b else 'pending'}
            if a and b and (a['status'] == b['status'] == 'completed'):
                for metric in metrics:
                    if a.get(metric) is not None and b.get(metric) is not None:
                        row[metric + '_orthogonal_minus_xavier'] = b[metric] - a[metric]
            paired.append(row)
    atomic_csv(out / 'run_status.csv', status_rows)
    atomic_csv(out / 'summary.csv', summary)
    atomic_csv(out / 'paired_differences.csv', paired)
    completed = sum((r['status'] == 'completed' for r in results))
    return (completed, len(results) - completed)
PROTOCOL_NOTE = 'MNIST BN35 paired initialization comparison.\nEight activations, two initializers, five seeds.\nOnly hidden-weight initialization differs between paired conditions.\nBoth arms share the data split, optimizer, dropout, stopping, and checkpoint rules.\nCompleted runs are skipped on restart; incomplete runs resume from committed checkpoints.\nNumerical failures remain recorded.\n'

def run_experiment():
    global _BUDGET
    args = make_args()
    if RUN_MODE not in ('all', 'screen_only', 'summary_only'):
        raise ValueError('RUN_MODE must be all, screen_only, or summary_only')
    if len(SEEDS) != 5 or len(set(SEEDS)) != 5:
        raise ValueError('This experiment requires five distinct seeds.')
    if len(set(ACTIVATION_PANEL)) != len(ACTIVATION_PANEL) or not set(ACTIVATION_PANEL) <= set(ALL_ACTIVATIONS):
        raise ValueError('Invalid/repeated activation names')
    if INITIALIZERS != ['xavier_normal', 'orthogonal_matched']:
        raise ValueError('Keep the two prespecified initialization conditions.')
    if not (1 <= args.min_epochs <= args.epochs and args.batch_size >= 2 and (args.workers == 0) and (args.depth > 0) and (args.width > 0) and (args.patience > 0) and (args.screen_steps > 0) and math.isfinite(args.lr) and (args.lr > 0) and (0 <= args.dropout < 1)):
        raise ValueError('Invalid protocol settings; notebook uses NUM_WORKERS=0.')
    if MAX_TOTAL_HOURS is not None and (not math.isfinite(MAX_TOTAL_HOURS) or MAX_TOTAL_HOURS <= 0):
        raise ValueError('MAX_TOTAL_HOURS must be positive or None')
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('Restart kernel and run this cell first to configure deterministic CUDA.')
    torch.set_default_dtype(torch.float32)
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device('cpu')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    with directory_lock(args.out):
        if RUN_MODE == 'summary_only':
            if not (args.out / 'manifest.json').exists():
                raise ValueError('No experiment in OUTPUT_DIR')
            print('Completed, numerical failures:', aggregate(args.out))
            return
        ensure_manifest(args)
        _BANK_CACHE.clear()
        _BUDGET = ActiveBudget(args.out / 'active_time.json', MAX_TOTAL_HOURS)
        (args.out / 'PROTOCOL.txt').write_text(PROTOCOL_NOTE, encoding='utf-8')
        total = len(ACTIVATION_PANEL) * len(INITIALIZERS) * len(SEEDS)
        print(f'{total} formal runs: {len(ACTIVATION_PANEL)} activations x 2 initializers x 5 seeds.')
        print(f'Device: {args.device}; output: {args.out}')
        print(f'Active time already used: {_BUDGET.previous / 3600:.2f} h; cap: {MAX_TOTAL_HOURS} h.')
        print('Interrupt normally to pause. Rerun this SAME cell to resume. No old results are imported.', flush=True)
        try:
            budget_check()
            risk = activation_risk(ACTIVATION_PANEL)
            atomic_csv(args.out / 'activation_risk.csv', risk)
            if any((not r['formula_finite'] for r in risk)):
                raise RuntimeError('Formula preflight failed; inspect activation_risk.csv before training.')
            data = Data(args, args.out)
            screening = []
            for r in risk:
                for method in ['orthogonal_matched', 'xavier_normal']:
                    screen = screen_one(args, data, r['activation'], method, SEEDS[0], args.out)
                    screening.append(screen)
                    atomic_csv(args.out / 'screening.csv', screening)
                    if screen['status'] != 'ok' and HALT_ON_NONFINITE:
                        print('HALTED on recorded screening failure. Inspect screening.csv. No hyperparameters have been changed.')
                        return
            if RUN_MODE == 'screen_only':
                print("Screen complete; not a guarantee of later stability. Set RUN_MODE='all' to continue.")
                return
            scores = {}
            for r in screening:
                score = (r['status'] != 'ok', r.get('max_abs_gradient', 0.0))
                scores[r['activation']] = max(scores.get(r['activation'], (False, 0.0)), score)
            order = sorted(ACTIVATION_PANEL, key=lambda a: scores[a], reverse=True)
            plan = []
            for seed in SEEDS:
                for activation in order:
                    methods = sorted([r for r in screening if r['activation'] == activation], key=lambda r: (r['status'] != 'ok', r.get('max_abs_gradient', 0.0)), reverse=True)
                    for r in methods:
                        plan.append({'order': len(plan) + 1, 'activation': activation, 'initializer': r['initializer'], 'seed': seed, 'run_id': run_id(activation, r['initializer'], seed)})
            atomic_csv(args.out / 'run_plan.csv', plan)
            aggregate(args.out)
            for spec in plan:
                budget_check()
                result = train_run(args, data, spec['activation'], spec['initializer'], spec['seed'], args.out)
                done, failed = aggregate(args.out)
                print(f'{done}/{total} completed; {failed} numerical failures; active time {_BUDGET.elapsed() / 3600:.2f} h', flush=True)
                if result['status'] != 'completed' and HALT_ON_NONFINITE:
                    print("HALTED on recorded numerical failure. See the run's failure.json. Other results are preserved.")
                    return
            print('ALL CONDITIONS PROCESSED. Check five-seed completeness and failure counts in summary.csv.')
        except KeyboardInterrupt:
            print('\nPAUSED. Completed runs and the last complete epoch are saved. Rerun the same cell to continue; the unfinished epoch will repeat.', flush=True)
        except BudgetStop as exc:
            print(f'\nBUDGET PAUSE: {exc} Remaining runs stay incomplete. Increase MAX_TOTAL_HOURS only if you deliberately allocate more time.', flush=True)
        finally:
            _BUDGET.flush()
            aggregate(args.out)
            _BUDGET = None
            _BANK_CACHE.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
IMPLEMENTATION_OBJECTS = ['Activation', 'DeepMLP', 'seed_all', 'NumericalFailure', 'check_tensor', 'check_named_tensors', 'forward_guards', 'atomic_json', 'atomic_csv', 'atomic_checkpoint', 'read_json', 'state_copy', 'worker_seed', 'Data', 'train_epoch', 'evaluate', 'diagnostics', 'optional_diagnostics', 'activation_risk', 'condition_id', 'run_id', 'screen_one', 'train_run', 'BudgetStop', 'ActiveBudget', 'budget_check', 'budget_flush', 'directory_lock', 'orthogonalize', 'file_sha256', 'initialization_bank', 'build_model', 'normalized_code', 'implementation_hash', 'make_args', 'ensure_manifest', 'aggregate', 'run_experiment']
run_experiment()
