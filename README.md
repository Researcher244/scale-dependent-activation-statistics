# Scale-dependent activation statistics

Training scripts, experimental records, and analysis code for the study of activation statistics and learning performance in deep neural networks.

The original panel contains 22 constructed and eight literature activations. The additional panel contains Logish, Smish, Gish, S-Mish, and CoLU. ReLU is included as a reference in the ResNet-32 experiment. Panel membership is recorded in `data/function_panels.csv`.

## Reproduce the analysis

Open `Reproduce_Results.ipynb` from the repository root and run its cells in order. This uses the included CSVs and does not train any networks. Install the analysis dependencies in the notebook environment with:

```python
%pip install -r requirements-analysis.txt
```

The notebook regenerates per-run and per-function summaries, rank correlations, the Fashion-MNIST no-BN bootstrap and ten-setting permutation calculations, Gaussian descriptors, and the five PDF figures. Analysis tables are written to `analysis/generated/`; plots are written to `figures/`. The four original-panel plots retain their original nine-setting layout; the later Fashion-MNIST no-BN experiment and additional-function panel are included in the tabular analysis.

`analysis/top_performers/reproduce_tables.py` separately reproduces the top-performing-function analysis from its accompanying input tables.

## Run training

Install a compatible PyTorch/torchvision build for the machine's CUDA version, then install `requirements-training.txt`. Open `Run_Training.ipynb`, select one script, and run the cells. Scripts download the datasets when needed. Keep the repository in a writable directory; on Kaggle, use `/kaggle/working/`.

MNIST and Fashion-MNIST runners support a local single-GPU environment. The CIFAR runners retain their Kaggle/Linux multi-GPU configuration; run these on that environment. The ResNet launchers invoke their accompanying worker files automatically.

| Experiment | Original panel | Additional five functions |
|---|---|---|
| MNIST BN, depth 10 | `training/mnist_bn10_original30.py` | `training/mnist_bn10_additional5.py` |
| MNIST BN, depth 35 | `training/mnist_bn35_original30.py` | `training/mnist_bn35_additional5.py` |
| MNIST no BN, depth 10 | `training/mnist_nobn10_original30.py` | `training/mnist_nobn10_additional5.py` |
| MNIST no BN, depth 15 | `training/mnist_nobn15_original30.py` | `training/mnist_nobn15_additional5.py` |
| Fashion-MNIST BN, depth 35 | `training/fashionmnist_bn35_original30.py` | `training/fashionmnist_bn35_additional5.py` |
| Fashion-MNIST no BN, depth 10 | `training/fashionmnist_nobn10_all35.py` (all 35) | Same runner |
| CIFAR-10 plain CNN BN, depths 30 and 40 | `training/cifar_bn30_bn40_original30.py` | `training/cifar_bn30_bn40_additional5.py` |
| CIFAR-10 plain CNN no BN, depth 15 | `training/cifar_nobn15_original30.py` | `training/cifar_nobn15_additional5.py` |
| CIFAR-10 ResNet-32 | `training/cifar_resnet32_original31.py` (includes ReLU) | `training/cifar_resnet32_additional5.py` |

The paired initialization experiment is `training/mnist_bn35_initialization.py`: eight activations, two initialization methods, and five seeds. The layerwise diagnostic experiments are `training/diagnostic_mnist_bn35.py` and `training/diagnostic_mnist_nobn15.py`; they use `training/layerwise_diagnostics.py`.

Each runner contains its architecture, optimizer, split, seed, stopping, and checkpoint settings. Outputs are saved under `runs/`. Resume by rerunning the same script with its output directory preserved; completed-run skipping and within-run checkpoint support follow each runner's implementation. Retain the same script version when resuming a checkpoint. For local Windows/Jupyter execution, use zero DataLoader workers. The supplied launcher sets this for the configurable MNIST runners, and the Fashion-MNIST no-BN runner already uses zero workers.

## Data and conventions

- `data/results/`: final run records, including test metrics where recorded.
- `data/epochs/`: epoch records, including validation metrics and available training metrics.
- `data/experiment_manifest.csv`: experiment keys, budgets, and paths to the corresponding tables.
- `data/function_panels.csv`: activation names and panel membership.
- `data/reference/`: fixed descriptor values, constructed-function parameters, and reference statistical calculations.
- `diagnostics/data/`: layerwise measurements for the diagnostic figure.

The result and epoch tables use `run_key` to identify runs; epoch tables add `epoch`. `cohort` distinguishes the original and additional panels, while `panel` distinguishes constructed, literature, additional, and ReLU-reference functions. The `initializer` field separates initialization conditions. Performance experiments use seeds 42–46; layerwise diagnostics use seeds 42 and 43.

The analysis evaluates fixed training-budget fractions (10%, 20%, and 50%), best observed validation loss, and recorded test outcomes. Unobserved epochs are not filled in after early stopping. Diverged runs remain in the exports; continuous correlations require finite outcomes, and the reported sample counts reflect that filtering. Loss correlations use negative loss, so positive correlations indicate better performance.

`analysis/compute_descriptors.py` evaluates the Gaussian statistics from explicit activation and derivative formulas. Rank-table reproduction uses the fixed descriptor tables in `data/reference/`.

Raw datasets, model checkpoints, and newly generated training outputs are excluded from version control. Dependency files list the required packages; `analysis/show_environment.py` prints the installed package versions and platform.
