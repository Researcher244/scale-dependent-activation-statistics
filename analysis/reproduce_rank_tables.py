"""Regenerate stage summaries and rank correlations from the final exports.

Run from Jupyter with %run analysis/reproduce_rank_tables.py.
No training, scale search, or predictor selection is performed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'generated'
OUT.mkdir(exist_ok=True)
manifest = pd.read_csv(ROOT / 'data/experiment_manifest.csv')
original = pd.read_csv(ROOT / 'data/reference/original_panel_descriptors.csv', index_col=0)
additional = pd.read_csv(ROOT / 'data/reference/additional_five_descriptors.csv', index_col=0)
pred = pd.concat([original, additional]).round(9)

rows = []
for entry in manifest.itertuples():
    results = pd.read_csv(ROOT / entry.results_file, low_memory=False, float_precision='round_trip')
    epochs = pd.read_csv(ROOT / entry.epochs_file, low_memory=False, float_precision='round_trip')
    assert not results.run_key.duplicated().any()
    assert not epochs.duplicated(['run_key', 'epoch']).any()
    grouped = {key: g.sort_values('epoch') for key, g in epochs.groupby('run_key')}
    for r in results.to_dict('records'):
        g = grouped[r['run_key']]
        z = {key: r[key] for key in ['experiment','run_key','activation','panel','cohort','initializer','seed']}
        z.update(test_acc=r.get('test_acc', np.nan), test_loss=r.get('test_loss', np.nan),
                 best_loss=g.val_loss.min(), stopped_at=r.get('stopped_at', np.nan),
                 selected_best_epoch=r.get('best_epoch', np.nan))
        if not np.isfinite(z['test_acc']):
            # Failed runs are retained in the data, but their partial losses do
            # not enter continuous finite-run correlations.
            z['best_loss'] = np.nan
        for fraction in [10,20,50]:
            q = g[g.epoch == entry.budget_epochs * fraction // 100]
            for metric in ['val_loss','val_acc','train_loss','train_acc']:
                z[f'{metric}{fraction}'] = q.iloc[0].get(metric, np.nan) if len(q)==1 else np.nan
        rows.append(z)
runs = pd.DataFrame(rows)
assert len(runs)==1835
runs.to_csv(OUT/'run_outcomes.csv', index=False)

keys=['experiment','initializer','panel','activation']
metrics=['test_acc','test_loss','best_loss']+[f'{m}{q}' for q in [10,20,50] for m in ['val_loss','val_acc','train_loss','train_acc']]
means=runs.groupby(keys,sort=True)[metrics].mean()
std=runs.groupby(keys,sort=True)[metrics].std()
counts=runs.groupby(keys,sort=True)[metrics].count()
summary=means.join(std.add_suffix('_sd')).join(counts.add_suffix('_n')).reset_index()
summary.to_csv(OUT/'function_summary.csv',index=False)

correlations=[]
for (ex,init,panel),g in summary.groupby(['experiment','initializer','panel']):
    if panel=='resnet_reference':continue
    g=g.set_index('activation')
    for metric in metrics:
        good=g[g[metric+'_n']==5]
        if len(good)<3:continue
        y=good[metric].round(12)*(-1 if 'loss' in metric else 1)
        for candidate in pred.columns:
            x=pred.loc[good.index,candidate]
            if x.isna().any():continue
            variable=x.nunique()>1 and y.nunique()>1
            correlations.append(dict(experiment=ex,initializer=init,panel=panel,
                outcome=metric,predictor=candidate,n_functions=len(good),
                spearman=spearmanr(x,y).statistic if variable else np.nan,
                kendall=kendalltau(x,y).statistic if variable else np.nan))
corr=pd.DataFrame(correlations)
corr.to_csv(OUT/'correlations.csv',index=False)

# Explicit additional-function ranks use means, average ties, and fixed score
# directions. Accuracy and loss ranks are intentionally separate.
new=summary[summary.panel=='additional5'].copy()
for metric in ['test_acc','best_loss','val_loss10','val_loss20','val_loss50']:
    new[metric+'_rank']=new.groupby('experiment')[metric].rank(ascending='loss' in metric,method='average')
for k in ['A25','RA','Gamma25','rho','B25','B1']:
    ranks=additional[k].round(9).rank(ascending=False,method='average')
    new[k+'_predicted_rank']=new.activation.map(ranks)
new.to_csv(OUT/'additional_function_rankings.csv',index=False)

# Verify against the supplied calculations used in the manuscript update.
labels={'MNIST BN10':'mnist_bn10','MNIST BN35':'mnist_bn35',
        'Fashion BN35':'fashionmnist_bn35','CIFAR BN30':'cifar10_bn30',
        'CIFAR BN40':'cifar10_bn40','CIFAR ResNet32':'cifar10_resnet32',
        'MNIST no-BN10':'mnist_nonorm10','MNIST no-BN15':'mnist_nonorm15',
        'CIFAR no-BN15':'cifar10_nonorm15','Fashion no-BN10':'fashionmnist_nonorm10'}
reference=pd.read_csv(ROOT/'data/reference/additional_five_correlations.csv')
reference['experiment']=reference.setting.map(labels)
check=reference.merge(corr[corr.panel=='additional5'],on=['experiment','predictor','outcome'],suffixes=('_reference','_regenerated'),validate='one_to_one')
assert len(check)==len(reference)
assert np.allclose(check.spearman_reference,check.spearman_regenerated,atol=1e-10)
assert np.allclose(check.kendall_reference,check.kendall_regenerated,atol=1e-10)
print(f'Validated {len(reference)} additional-function correlation rows against manuscript calculations.')
print('Wrote run outcomes, function summaries, correlations, and explicit ranks to',OUT)
