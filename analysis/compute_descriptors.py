"""Recompute the paper's fixed descriptors; this does not select new scales."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from activation_math import fd, params

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/generated'
OUT.mkdir(exist_ok=True)
panels=pd.read_csv(ROOT/'data/function_panels.csv')
names=panels.loc[panels.panel!='resnet_reference','activation'].tolist()
scales=[.05,.1,.2,.25,1.,3.,3.85,5.25]
rows=[]
for name in names:
    x=np.linspace(-8,8,321)
    f,d=fd(name,x)
    h=1e-4
    numeric=(-fd(name,x+2*h)[0]+8*fd(name,x+h)[0]-8*fd(name,x-h)[0]+fd(name,x-2*h)[0])/(12*h)
    assert np.allclose(d,numeric,atol=3e-8),name
    cache={}
    for s in scales:
        def integrand(z):
            y,dy=fd(name,s*z)
            return np.array([dy,dy*dy,y,y*y])*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        v,error=quad_vec(integrand,-12,12,points=[-1,0,1],epsabs=2e-11,epsrel=2e-11)
        cache[s]={'A':v[0],'B':v[1],'M':v[3],'V':v[3]-v[2]**2}
    # Impose analytic constant-A identities before ranking.
    p=params.get(name,{})
    if name in ['Swish','GeluAct'] or 'scale' in p or (p and p.get('b',0)==0 and p.get('alpha',0)==0 and p.get('gamma',p.get('eta',0))==0):
        for c in cache.values():c['A']=.5*p.get('scale',1.)
    a=lambda s:cache[s]['A']
    b=lambda s:cache[s]['B']
    v=lambda s:cache[s]['V']
    rows.append(dict(activation=name,RA=a(.25)/a(3.),RB=b(.25)/b(3.),
        Gamma25=a(.25)/np.sqrt(v(1.)),rho=a(1.)**2/v(1.),A25=a(.25),
        c1=a(1.),B25=b(.25),B1=b(1.),V1=v(1.),M1=cache[1.]['M'],
        slope0=float(fd(name,0.)[1]),GammaLoc25=.25*a(.25)/np.sqrt(v(.25)),
        A05=a(.05),RAstar=a(.2)/a(5.25),RBstar=b(.1)/b(3.85)))
df=pd.DataFrame(rows).set_index('activation').round(9)
for filename in ['original_panel_descriptors.csv','additional_five_descriptors.csv']:
    ref=pd.read_csv(ROOT/'data/reference'/filename,index_col=0)
    cols=ref.columns.intersection(df.columns)
    err=(df.loc[ref.index,cols]-ref[cols]).abs().max().max()
    assert err<1e-7,(filename,err)
    print(f'{filename}: maximum absolute difference {err:.2g}')
df.to_csv(OUT/'recomputed_descriptors.csv')
print('Verified all analytic derivatives and fixed descriptor values against the reference tables.')
print('Wrote',OUT/'recomputed_descriptors.csv')
