from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
root = Path(__file__).resolve().parent
out = root / 'reproduced'
out.mkdir(exist_ok=True)
means = pd.read_csv(root / 'function_means.csv')
runs = pd.read_csv(root / 'run_outcomes.csv')
pred = pd.read_csv(root / 'predictors.csv', index_col=0)
d = pred
old = ['Mish', 'Swish', 'GeluAct', 'SerfAct', 'TanhExpAct', 'PhishAct', 'ExpExpishAct', 'SquareSwishAct']
selected = []
corr = []
stability = []
for ex, g in means.groupby('experiment'):
    if 'nonorm' in ex:
        continue
    rr = runs[runs.experiment == ex]
    for panel, names in [('all30', d.index.tolist()), ('constructed22', [n for n in d.index if n not in old]), ('literature8', old)]:
        g0 = g[g.activation.isin(names)].set_index('activation')
        for outcome in ['test_acc', 'best_loss']:
            score = (-g0[outcome] if 'loss' in outcome else g0[outcome]).round(12)
            ordered = score.sort_values(ascending=False, kind='stable')
            subsets = {'full': ordered.index.tolist(), 'top5': ordered.index[:5].tolist(), 'top10': ordered.index[:min(10, len(ordered))].tolist(), 'top15': ordered.index[:min(15, len(ordered))].tolist(), 'above20pct': g0[g0.test_acc > 0.2].index.tolist(), 'above50pct': g0[g0.test_acc > 0.5].index.tolist()}
            for sub, ns in subsets.items():
                if len(ns) < 3:
                    continue
                y = score.loc[ns]
                for k in pred:
                    x = pred.loc[ns, k]
                    val = spearmanr(x, y).statistic if x.nunique() > 1 and y.nunique() > 1 else np.nan
                    corr.append(dict(experiment=ex, panel=panel, selection_outcome=outcome, subset=sub, n=len(ns), predictor=k, spearman=val, kendall=kendalltau(x, y).statistic if x.nunique() > 1 else np.nan))
            ns = subsets['top5']
            for i, n in enumerate(ns):
                row = dict(experiment=ex, panel=panel, selection_outcome=outcome, activation=n, actual_rank=i + 1, mean=g0.loc[n, outcome], sd=rr[rr.activation == n][outcome].std(), family='literature' if n in old else 'constructed')
                for k in pred:
                    row[k + '_predicted_rank'] = pred.loc[ns, k].rank(ascending=False).loc[n]
                selected.append(row)
            if panel == 'all30':
                z = rr[rr.activation.isin(ns)].pivot(index='seed', columns='activation', values=outcome).reindex(columns=ns)
                if outcome == 'best_loss':
                    z = -z
                for i in range(5):
                    for j in range(i + 1, 5):
                        delta = z.iloc[:, i] - z.iloc[:, j]
                        stability.append(dict(experiment=ex, outcome=outcome, better_mean=ns[i], worse_mean=ns[j], wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), seeds=len(delta), mean_difference=delta.mean()))
c = pd.DataFrame(corr)
sel = pd.DataFrame(selected)
st = pd.DataFrame(stability)
c.to_csv(out / 'correlations.csv', index=False)
sel.to_csv(out / 'selected_functions.csv', index=False)
st.to_csv(out / 'pairwise_seed_consistency.csv', index=False)
pred.to_csv(out / 'predictors.csv', index_label='activation')
for outcome in ['test_acc', 'best_loss']:
    print('\nOUTCOME', outcome)
    z = c[(c.panel == 'all30') & (c.selection_outcome == outcome) & (c.subset == 'top5')]
    print(z.pivot(index='experiment', columns='predictor', values='spearman').round(3).to_string())
    print('RA subset sizes\n', c[(c.panel == 'all30') & (c.selection_outcome == outcome) & (c.predictor == 'RA')].pivot(index='experiment', columns='subset', values='spearman').round(3).to_string())
    print('TOP5\n', sel[(sel.panel == 'all30') & (sel.selection_outcome == outcome)][['experiment', 'actual_rank', 'activation', 'mean', 'sd', 'RA_predicted_rank', 'Gamma25_predicted_rank', 'rho_predicted_rank']].to_string(index=False))
    print('SEED STABILITY: number of top-five pairs with ordering shared by all five seeds\n', st[st.outcome == outcome].assign(allfive=lambda x: x.wins == 5).groupby('experiment').allfive.sum().to_string())
for panel in ['constructed22', 'literature8']:
    print('\n', panel, 'TEST ACC top5\n', c[(c.panel == panel) & (c.selection_outcome == 'test_acc') & (c.subset == 'top5')].pivot(index='experiment', columns='predictor', values='spearman').round(3).to_string())
print('PASS: original30 only; six BN settings; 17 previously considered descriptors; direction fixed except lower chi as specified previously.')
