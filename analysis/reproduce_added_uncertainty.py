"""Fashion-MNIST no-BN bootstrap and ten-setting permutation/Holm calculations."""
from pathlib import Path
import itertools, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/generated'
if not (OUT/'run_outcomes.csv').exists():
    raise FileNotFoundError('Run analysis/reproduce_rank_tables.py first.')
runs=pd.read_csv(OUT/'run_outcomes.csv')
pred=pd.read_csv(ROOT/'data/reference/original_panel_descriptors.csv',index_col=0).round(9)
panel=pd.read_csv(ROOT/'data/function_panels.csv').set_index('activation').panel
constructed=[n for n in pred.index if panel[n]=='constructed22']
fr=runs[runs.experiment=='fashionmnist_nonorm10']
y=fr[fr.activation.isin(constructed)].pivot(index='activation',columns='seed',values='test_acc').reindex(constructed).to_numpy()
assert y.shape==(22,5)

def rowscorr(x,y):
    x=rankdata(x,axis=1);y=rankdata(y,axis=1)
    x-=x.mean(1,keepdims=True);y-=y.mean(1,keepdims=True)
    with np.errstate(invalid='ignore',divide='ignore'):
        return np.sum(x*y,1)/np.sqrt(np.sum(x*x,1)*np.sum(y*y,1))

rng=np.random.default_rng(20261101)
fi=rng.integers(0,22,(10000,22));si=rng.integers(0,5,(10000,5))
by=np.round(y[fi[:,:,None],si[:,None,:]].mean(2),12)
ra=rowscorr(pred.loc[constructed,'RA'].to_numpy()[fi],by)
aa=rowscorr(pred.loc[constructed,'A25'].to_numpy()[fi],by)
shared=np.array(list(itertools.product(range(5),repeat=5)))
sy=np.round(y[:,shared].mean(2).T,12)
sr=rowscorr(np.broadcast_to(pred.loc[constructed,'RA'].to_numpy(),sy.shape),sy)
unc={'joint_ra':np.nanquantile(ra,[.025,.975]).tolist(),
     'joint_delta':np.nanquantile(ra-aa,[.025,.975]).tolist(),
     'seed_ra':np.nanquantile(sr,[.025,.975]).tolist(),
     'undefined':int(np.isnan(ra).sum()),
     'tau':float(kendalltau(pred.loc[constructed,'RA'],y.mean(1).round(12)).statistic)}
ref=json.loads((ROOT/'data/reference/fashion_noBN_uncertainty.json').read_text())
for k in ref:assert np.allclose(unc[k],ref[k],atol=1e-10),(k,unc[k],ref[k])
(OUT/'fashion_noBN_uncertainty.json').write_text(json.dumps(unc,indent=2)+'\n')

pv=[]
eligible=runs[runs.activation.isin(constructed)&(runs.experiment!='mnist_bn35_init')]
for ex,g in eligible.groupby('experiment'):
    z=g.pivot(index='activation',columns='seed',values='test_acc').dropna().sort_index()
    x=pred.loc[z.index,'RA'].to_numpy();yy=np.round(z.mean(1).to_numpy(),12)
    obs=float(spearmanr(x,yy).statistic)
    xr=rankdata(x);xr-=xr.mean();yr=rankdata(yy);yr-=yr.mean()
    permutations=np.array([rng.permutation(yr) for _ in range(49999)])
    vals=permutations@xr/np.sqrt(np.sum(xr*xr)*np.sum(yr*yr))
    pv.append({'experiment':ex,'spearman':obs,'p':(1+int((np.abs(vals)>=abs(obs)-1e-12).sum()))/50000})
pv=pd.DataFrame(pv).sort_values('p')
pv['holm']=np.minimum(1,np.maximum.accumulate(pv.p.to_numpy()*np.arange(len(pv),0,-1)))
reference=pd.read_csv(ROOT/'data/reference/permutation_ten_settings.csv').set_index('experiment').sort_index()
check=pv.set_index('experiment').sort_index()
assert np.allclose(check[reference.columns],reference,atol=1e-10)
pv.to_csv(OUT/'permutation_ten_settings.csv',index=False)
print('Verified added bootstrap intervals and all ten permutation/Holm rows against the reference calculations.')
