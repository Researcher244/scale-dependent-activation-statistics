"""CIFAR-10 ResNet-32; five additional functions.

Five seeds per activation; source training and resume rules retained.
"""

from pathlib import Path
import os, sys, shutil, subprocess, signal
WORK_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
OUTPUT_DIR = __import__('pathlib').Path(__file__).resolve().parents[1] / 'runs/cifar_resnet32_additional5'
DATA_DIR = __import__('pathlib').Path(__file__).resolve().parents[1] / 'datasets'
RESTORE_FROM = None
NUM_WORKERS = 2
GPUS = 2
SESSION_HOURS = 0
SCRIPT = Path(__file__).with_name("cifar_resnet32_additional5_worker.py").read_text(encoding="utf-8")
if 'NEW_FUNCTIONS_ONLY = False' in SCRIPT:
    OUTPUT_DIR = WORK_DIR / 'CIFAR_ResNet32_ExpandedPanel'
if RESTORE_FROM:
    restore = Path(RESTORE_FROM)
    if not restore.is_dir():
        raise FileNotFoundError(f'Restore folder not found: {restore}')
    if not OUTPUT_DIR.exists():
        shutil.copytree(restore, OUTPUT_DIR)
        print(f'Restored previous run folder to {OUTPUT_DIR}', flush=True)
    else:
        print('Using existing writable output folder; restore source will not overwrite it.', flush=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
script_path = WORK_DIR / 'cifar_resnet32_new_functions.py'
script_path.write_text(SCRIPT.lstrip('\n'), encoding='utf-8')
command = [sys.executable, '-u', str(script_path), '--out', str(OUTPUT_DIR), '--data', str(DATA_DIR), '--workers', str(NUM_WORKERS), '--gpus', str(GPUS), '--session-hours', str(SESSION_HOURS)]
env = dict(os.environ, PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
print('Starting CIFAR ResNet-32; the worker PLAN reports the selected run count. Completed runs are skipped.', flush=True)
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, start_new_session=True)
interrupted = False
try:
    for line in process.stdout:
        print(line, end='', flush=True)
    returncode = process.wait()
except KeyboardInterrupt:
    interrupted = True
    print('Requesting a checkpoint-safe pause. Please allow the workers to exit.', flush=True)
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        print('Workers terminated. Last completed epoch checkpoints remain available.', flush=True)
finally:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
if not interrupted and process.returncode != 0:
    raise RuntimeError(f'Experiment stopped with exit code {process.returncode}; inspect the output above.')
print(f'Output folder: {OUTPUT_DIR}')
print('Rerun this cell to resume. Preserve the entire folder when moving Kaggle sessions.')
