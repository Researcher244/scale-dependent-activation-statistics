"""Generate the four original-panel manuscript figures from the final CSVs.
The additional-five cohort is evaluated separately in the numerical tables.
"""

from pathlib import Path
from io import StringIO
import warnings
import math
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad
from scipy.special import expit, erf, ndtr
from scipy.stats import spearmanr, kendalltau
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
OUTPUT_DIR = DATA_DIR.parent / 'figures'
SAVE_SVG = True
SAVE_PNG = False
PNG_DPI = 300
PANEL = 'constructed'
SEEDS = [42, 43, 44, 45, 46]
FONT_SIZE = 9
EXPERIMENTS = {'mnist_bn10': ('MNIST BN, depth 10', 20), 'mnist_bn35': ('MNIST BN, depth 35', 100), 'cifar10_bn30': ('CIFAR-10 BN, depth 30', 60), 'cifar10_bn40': ('CIFAR-10 BN, depth 40', 60), 'fashionmnist_bn35': ('Fashion-MNIST BN, depth 35', 100), 'mnist_nonorm10': ('MNIST no BN, depth 10', 20), 'mnist_nonorm15': ('MNIST no BN, depth 15', 100), 'cifar10_nonorm15': ('CIFAR-10 no BN, depth 15', 60), 'cifar10_resnet32': ('CIFAR-10 ResNet-32', 60)}
CONSTRUCTED = ['FixC1MatchSlopeAndRho', 'FixC1FixRhoMatchSlope', 'FixC1FixSlopeMatchRho', 'FixRhoMatchC1AndSlope', 'FixRhoFixSlopeMatchC1', 'FixSlopeMatchC1AndRho', 'FixSlopeandC1HighRho', 'FixSlopeandC1LowRho', 'MatchAllToMish', 'BandHigh', 'BandLow', 'GentleTiltDown09', 'ScaledSwish', 'HighBandSwish', 'LowBandSwish', 'SetD5_HighA_LowGamma', 'ScaledSwishhalf', 'SetD1_LowA_HighGamma', 'SetD2_LowA_LowGamma', 'SetD3_MidA_LowGamma', 'SetD4_MidA_HighGamma', 'SetD6_HighA_HighGamma']
LITERATURE = ['Mish', 'Swish', 'GeluAct', 'SerfAct', 'TanhExpAct', 'PhishAct', 'ExpExpishAct', 'SquareSwishAct']
RA = 'A(0.25)/A(3)'
RB = 'B(0.25)/B(3)'
PRIMARY_PREDICTOR = RA
STAT_STYLES = {'A(0.25)': dict(color='#0072B2', marker='o', linestyle='-'), 'B(0.25)': dict(color='#E69F00', marker='s', linestyle='--'), 'GammaBN(0.25)': dict(color='#009E73', marker='^', linestyle='-.'), RA: dict(color='#D55E00', marker='D', linestyle='-'), RB: dict(color='#CC79A7', marker='v', linestyle='--'), 'B(1)': dict(color='#555555', marker='P', linestyle=':')}
STAT_LABELS = {'A(0.25)': '$A(0.25)$', 'B(0.25)': '$B(0.25)$', 'B(1)': '$B(1)$', 'GammaBN(0.25)': '$\\Gamma_{\\mathrm{BN}}(0.25)$', RA: '$R_A(0.25,3)$', RB: '$R_B(0.25,3)$'}
SHORT_NAMES = {'SetD1_LowA_HighGamma': 'D1', 'SetD2_LowA_LowGamma': 'D2', 'SetD3_MidA_LowGamma': 'D3', 'SetD4_MidA_HighGamma': 'D4', 'SetD5_HighA_LowGamma': 'D5', 'SetD6_HighA_HighGamma': 'D6', 'ScaledSwishhalf': '0.5 Swish', 'ScaledSwish': '1.4 Swish', 'GeluAct': 'GELU', 'SerfAct': 'SERF', 'TanhExpAct': 'TanhExp', 'ExpExpishAct': 'GoLU', 'SquareSwishAct': 'SquareSwish', 'PhishAct': 'Phish'}
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': FONT_SIZE, 'mathtext.fontset': 'dejavuserif', 'text.usetex': False, 'axes.titlesize': FONT_SIZE, 'axes.labelsize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE - 1, 'ytick.labelsize': FONT_SIZE - 1, 'legend.fontsize': FONT_SIZE - 1, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7, 'lines.linewidth': 1.6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if PANEL not in ['constructed', 'literature']:
    raise ValueError("PANEL must be 'constructed' or 'literature'.")

def finish_figure(fig, stem):
    fig.savefig(OUTPUT_DIR / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.06)
    if SAVE_SVG:
        fig.savefig(OUTPUT_DIR / f'{stem}.svg', bbox_inches='tight', pad_inches=0.06)
    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f'{stem}.png', dpi=PNG_DPI, bbox_inches='tight', pad_inches=0.06)
    print('Saved:', (OUTPUT_DIR / f'{stem}.pdf').resolve())
    plt.show()

def panel_title(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', pad=7)
print('Data directory:', DATA_DIR.resolve())
print('Figure directory:', OUTPUT_DIR.resolve())
print('Python', sys.version.split()[0], '| NumPy', np.__version__, '| pandas', pd.__version__, '| Matplotlib', matplotlib.__version__)
PREDICTOR_CSV = '\nactivation,A(0.25),B(0.25),GammaBN(0.25),A(0.25)/A(3),A(0.2)/A(5.25),B(0.25)/B(3),B(0.1)/B(3.85),A(0.05),slope0,c1,rho,variance,second_moment\nMish,0.596996483,0.379571908,0.950433553,1.163508319,1.187919072,0.735674102,0.706024737,0.599879966,0.6,0.563747557,0.805507003,0.394548163,0.452342192\nSwish,0.5,0.264714135,0.893593611,1.0,1.0,0.534921394,0.501504288,0.5,0.5,0.5,0.798509542,0.313083297,0.35577552\nFixC1MatchSlopeAndRho,0.584557962,0.354674828,1.049282032,1.191592183,1.186512285,0.727258336,0.717430085,0.599328156,0.6,0.5,0.805507003,0.310363534,0.34884019\nFixC1FixRhoMatchSlope,0.584433929,0.355421849,1.044492849,1.189894238,1.185768358,0.724255155,0.715127458,0.599322225,0.6,0.5,0.798509542,0.313083297,0.353804627\nFixC1FixSlopeMatchRho,0.5,0.263566252,0.89750042,1.0,1.0,0.534864592,0.50218831,0.5,0.5,0.5,0.805507003,0.310363534,0.350540162\nFixRhoMatchC1AndSlope,0.610096612,0.375972763,0.967061281,1.205824595,1.211633694,0.691738467,0.674338714,0.600559115,0.6,0.563747557,0.798509542,0.398005649,0.432247583\nFixRhoFixSlopeMatchC1,0.536269661,0.28987085,0.850038526,1.063783967,1.047222374,0.491043817,0.444313827,0.501652941,0.5,0.563747557,0.798509542,0.398005649,0.418131636\nFixSlopeMatchC1AndRho,0.533932362,0.287294813,0.850033872,1.057301147,1.043785555,0.497283079,0.45167863,0.501560991,0.5,0.563747557,0.805507003,0.394548163,0.415478053\nMatchAllToMish,0.604322779,0.371907136,0.962097202,1.190280556,1.20248048,0.697717286,0.683679847,0.600268932,0.6,0.563747557,0.805507003,0.394548163,0.433155637\nFixSlopeandC1HighRho,0.5,0.253846304,0.95334602,1.0,1.0,0.584354959,0.541651576,0.5,0.5,0.5,0.908868635,0.275067254,0.288996497\nFixSlopeandC1LowRho,0.5,0.300307985,0.847880728,1.0,1.0,0.590788807,0.510916652,0.5,0.5,0.5,0.71890173,0.347752676,0.439472467\nBandHigh,0.651259925,0.446171991,1.036898174,1.28695848,1.297291392,0.869506952,0.780836215,0.610582215,0.599999995,0.563705633,0.805505034,0.394490446,0.44918948\nBandLow,0.557788719,0.313398372,0.888094665,1.104003054,1.107268036,0.550649042,0.5814924,0.589156848,0.599999997,0.563693854,0.80550023,0.394476314,0.417310299\nGentleTiltDown09,0.608052372,0.393554674,0.96803581,1.193708903,1.196493388,0.76166595,0.618582501,0.525496719,0.507372802,0.563652184,0.805236143,0.394547347,0.449382351\nScaledSwish,0.7,0.518839704,0.893593611,1.0,1.0,0.534921394,0.501504288,0.7,0.7,0.7,0.798509542,0.613643262,0.697320019\nHighBandSwish,0.550016749,0.31802933,0.98291548,1.114800528,1.1007039,0.646888794,0.561908611,0.509872852,0.500008221,0.498920291,0.794955563,0.313126253,0.35584892\nLowBandSwish,0.450018047,0.21717555,0.804261142,0.887753644,0.899978124,0.431423995,0.443141305,0.490153995,0.500007078,0.501093372,0.80199494,0.313087471,0.353847954\nGeluAct,0.5,0.285356986,0.850463122,1.0,1.0,0.558171758,0.501563215,0.5,0.5,0.5,0.723287521,0.345644011,0.425221483\nTanhExpAct,0.742043273,0.586576211,1.093630707,1.433064446,1.484388121,1.11466587,1.116935814,0.760771523,0.761594156,0.614385603,0.819907741,0.460380662,0.529093704\nExpExpishAct,0.368666591,0.165923374,0.73348825,0.761519609,0.7424802,0.339407953,0.28180799,0.367880871,0.367879441,0.412635774,0.673988536,0.252627861,0.319634353\nSerfAct,0.666674407,0.471444492,1.005885746,1.287304789,1.325786111,0.900908019,0.879073695,0.67278838,0.67304129,0.599803199,0.819006402,0.439268699,0.501423121\nPhishAct,0.065817956,0.067812518,0.1302766,0.134718009,0.088561344,0.127137891,0.019402818,0.00297652,0.0,0.364707829,0.521115779,0.255244239,0.361053489\nSquareSwishAct,0.261143671,0.083135635,0.595100201,0.55308691,0.522123049,0.173172675,0.132316303,0.250467776,0.25,0.35577552,0.657314643,0.192565649,0.235257871\nSetD1_LowA_HighGamma,0.434988627,0.192698758,0.959717572,0.941617394,0.899709507,0.487959505,0.442206999,0.43915861,0.439666673,0.426895043,0.887101497,0.205432387,0.218153749\nScaledSwishhalf,0.25,0.066178534,0.893593611,1.0,1.0,0.534921394,0.501504288,0.25,0.25,0.25,0.798509542,0.078270824,0.08894388\nSetD5_HighA_LowGamma,0.615063805,0.379586174,0.820324264,0.801717219,0.880336038,0.588231182,0.552503147,0.583892226,0.580686427,0.746000735,0.989940938,0.562172019,0.563304109\nSetD2_LowA_LowGamma,0.445001524,0.200968442,0.800054907,0.80932193,0.838369235,0.432853419,0.380273963,0.424476329,0.422634043,0.539010708,0.939098996,0.309373714,0.318560905\nSetD3_MidA_LowGamma,0.525055346,0.277403064,0.810386948,0.812998314,0.88383905,0.529743761,0.479442089,0.497362055,0.494547821,0.640429775,0.977052131,0.419783431,0.423403191\nSetD4_MidA_HighGamma,0.534990881,0.29869428,0.939864663,1.065087688,1.070979366,0.6059722,0.580611905,0.539188801,0.539630885,0.516127595,0.822151857,0.324012763,0.361674858\nSetD6_HighA_HighGamma,0.625037648,0.392048778,0.900253987,0.96423247,1.071852553,0.726286342,0.691515889,0.607974978,0.606145866,0.687690716,0.981079216,0.482039078,0.486344327\n'
PARAMETER_CSV = '\nactivation,beta,b,alpha1,alpha2\nFixC1MatchSlopeAndRho,1.0056290824,-0.6108569031,0.2481362485,0.0000000000\nFixC1FixRhoMatchSlope,1.0655913532,-0.6847583639,0.2647998861,0.0000000000\nFixC1FixSlopeMatchRho,0.9579116739,0.0000000000,0.0000000000,0.0000000000\nFixRhoMatchC1AndSlope,2.5097657107,3.2396289680,-0.3622986508,0.0000000000\nFixRhoFixSlopeMatchC1,3.6454545499,5.5094463356,-0.4959679749,0.0000000000\nFixSlopeMatchC1AndRho,3.2799942834,4.9459659006,-0.4929381821,0.0000000000\nFixSlopeandC1HighRho,0.5000000000,0.0000000000,0.0000000000,0.0000000000\nFixSlopeandC1LowRho,2.0000000000,0.0000000000,0.0000000000,0.0000000000\nMatchAllToMish,2.0996330505,2.5444438979,-0.3271993629,0.0000000000\nHighBandSwish,-1.0559000000,-0.5061000000,0.1947000000,-0.0708000000\nLowBandSwish,-1.0127000000,0.4625000000,-0.1843000000,0.0707000000\nBandHigh,1.2068002800,0.0446194600,0.1656806700,-0.0768336900\nBandLow,3.0926678300,4.5683465600,-0.4735516900,0.0838202500\nGentleTiltDown09,1.3212061309,0.6706517594,-0.0277762821,-0.1265000000\nSetD1_LowA_HighGamma,0.4949478828,-0.4137562830,0.0388663503,0.0027885451\nSetD2_LowA_LowGamma,0.4208138600,0.4835911900,-0.1879102600,-0.0080512200\nSetD3_MidA_LowGamma,0.3093398248,1.0044482442,-0.2236286170,-0.0137558189\nSetD4_MidA_HighGamma,0.9163946020,0.0436458135,0.0264917301,0.0022294333\nSetD5_HighA_LowGamma,0.2500000000,1.6646126199,-0.2443422477,-0.0158275460\nSetD6_HighA_HighGamma,0.3558473158,1.1158205221,-0.1377719114,-0.0092948660\n'
B1_CSV = '\nactivation,B(1)\nMish,0.4790837583739699\nSwish,0.3794823516328294\nFixC1MatchSlopeAndRho,0.3762412901586505\nFixC1FixRhoMatchSlope,0.3820432089853285\nFixC1FixSlopeMatchRho,0.3735743167217198\nFixRhoMatchC1AndSlope,0.4892543563220614\nFixRhoFixSlopeMatchC1,0.5180761020663184\nFixSlopeMatchC1AndRho,0.5011471246751036\nMatchAllToMish,0.477400548268546\nFixSlopeandC1HighRho,0.300307985085066\nFixSlopeandC1LowRho,0.4664250548387369\nBandHigh,0.4814035139518056\nBandLow,0.4954356972424918\nGentleTiltDown09,0.479939906129984\nScaledSwish,0.7437854092003455\nHighBandSwish,0.3824024545998351\nLowBandSwish,0.3797912413543665\nGeluAct,0.4558508656492872\nTanhExpAct,0.5611713835579049\nExpExpishAct,0.3475001486002479\nSerfAct,0.5299807256192134\nPhishAct,0.4411801591616928\nSquareSwishAct,0.2657122013722097\nSetD1_LowA_HighGamma,0.2288372280073985\nScaledSwishhalf,0.0948705879082073\nSetD5_HighA_LowGamma,0.57467553958768\nSetD2_LowA_LowGamma,0.3318784107166164\nSetD3_MidA_LowGamma,0.4348144450696985\nSetD4_MidA_HighGamma,0.3841946662139698\nSetD6_HighA_HighGamma,0.4928826506720911\n'
predictors = pd.read_csv(StringIO(PREDICTOR_CSV)).set_index('activation')
parameters = pd.read_csv(StringIO(PARAMETER_CSV)).set_index('activation')
predictors['B(1)'] = pd.read_csv(StringIO(B1_CSV)).set_index('activation')['B(1)']

def activation(name, x):
    x = np.asarray(x, dtype=float)
    if name in parameters.index:
        p = parameters.loc[name]
        return x * expit(p.beta * x + p.b) + p.alpha1 * x * np.exp(-0.5 * x * x) + p.alpha2 * x * np.exp(-0.5 * (x / 0.15) ** 2)
    if name in ['Swish', 'ScaledSwish', 'ScaledSwishhalf']:
        scale = {'Swish': 1.0, 'ScaledSwish': 1.4, 'ScaledSwishhalf': 0.5}[name]
        return scale * x * expit(x)
    if name == 'Mish':
        return x * np.tanh(np.logaddexp(0.0, x))
    if name == 'GeluAct':
        return x * ndtr(x)
    if name == 'SerfAct':
        return x * erf(np.logaddexp(0.0, x))
    if name == 'TanhExpAct':
        return x * np.tanh(np.exp(np.minimum(x, 40)))
    if name == 'PhishAct':
        return x * np.tanh(x * ndtr(x))
    if name == 'ExpExpishAct':
        return x * np.exp(-np.exp(np.minimum(-x, 40)))
    if name == 'SquareSwishAct':
        return x * expit(x) ** 2
    if name == 'ReLU':
        return np.maximum(x, 0)
    raise KeyError(f'No definition for {name}')

def derivative(name, x):
    x = np.asarray(x, dtype=float)
    if name in parameters.index:
        p = parameters.loc[name]
        g = expit(p.beta * x + p.b)
        return g + p.beta * x * g * (1 - g) + p.alpha1 * np.exp(-0.5 * x * x) * (1 - x * x) + p.alpha2 * np.exp(-0.5 * (x / 0.15) ** 2) * (1 - (x / 0.15) ** 2)
    if name in ['Swish', 'ScaledSwish', 'ScaledSwishhalf']:
        scale = {'Swish': 1.0, 'ScaledSwish': 1.4, 'ScaledSwishhalf': 0.5}[name]
        g = expit(x)
        return scale * (g + x * g * (1 - g))
    if name == 'Mish':
        g = np.tanh(np.logaddexp(0.0, x))
        return g + x * (1 - g * g) * expit(x)
    if name == 'GeluAct':
        return ndtr(x) + x * np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    if name == 'SerfAct':
        sp = np.logaddexp(0.0, x)
        return erf(sp) + 2 * x * np.exp(-sp * sp) * expit(x) / np.sqrt(np.pi)
    if name == 'TanhExpAct':
        ex = np.exp(np.minimum(x, 40))
        g = np.tanh(ex)
        return g + x * ex * (1 - g * g)
    if name == 'PhishAct':
        g = np.tanh(x * ndtr(x))
        dg = ndtr(x) + x * np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
        return g + x * (1 - g * g) * dg
    if name == 'ExpExpishAct':
        ex = np.exp(np.minimum(-x, 40))
        g = np.exp(-ex)
        return g + x * ex * g
    if name == 'SquareSwishAct':
        g = expit(x)
        return g * g + 2 * x * g * g * (1 - g)
    if name == 'ReLU':
        return np.where(x == 0, np.nan, (x > 0).astype(float))
    raise KeyError(name)

def gaussian_expectation(fun):
    return quad(lambda z: float(fun(z)) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi), -12, 12, points=[-0.6, -0.15, 0, 0.15, 0.6], epsabs=2e-11, epsrel=2e-11, limit=250)[0]

def gaussian_A(name, s):
    if s < 0:
        raise ValueError('s is a standard deviation; use s >= 0.')
    if s == 0:
        return float(derivative(name, 0.0))
    if name in ['Swish', 'GeluAct', 'ScaledSwish', 'ScaledSwishhalf']:
        return {'Swish': 0.5, 'GeluAct': 0.5, 'ScaledSwish': 0.7, 'ScaledSwishhalf': 0.25}[name]
    if name in parameters.index:
        p = parameters.loc[name]
        if p.b == 0 and p.alpha1 == 0 and (p.alpha2 == 0):
            return 0.5
    return gaussian_expectation(lambda z: derivative(name, s * z))
for name in predictors.index:
    mu = gaussian_expectation(lambda z: activation(name, 0.25 * z))
    m2 = gaussian_expectation(lambda z: activation(name, 0.25 * z) ** 2)
    predictors.loc[name, 'GammaLoc(0.25)'] = 0.25 * predictors.loc[name, 'A(0.25)'] / np.sqrt(m2 - mu * mu)
predictors = predictors.round(9)
assert set(predictors.index) == set(CONSTRUCTED + LITERATURE)
assert np.isfinite(predictors.to_numpy()).all()
predictors.to_csv(OUTPUT_DIR / 'predictors_used.csv')
print(f'{len(CONSTRUCTED)} constructed + {len(LITERATURE)} literature functions; {len(predictors.columns)} descriptors available.')

def locate_csv(filename):
    for suffix, folder in [('_results.csv', 'results'), ('_epochs.csv', 'epochs')]:
        if filename.endswith(suffix):
            path = DATA_DIR / folder / (filename[:-len(suffix)] + '.csv')
            if path.is_file():
                return path
    raise FileNotFoundError(filename)
results, epochs, audit_rows = ({}, {}, [])
for exp in list(EXPERIMENTS) + ['mnist_bn35_init']:
    try:
        rp = locate_csv(exp + '_results.csv')
        ep = locate_csv(exp + '_epochs.csv')
    except FileNotFoundError:
        if exp == 'mnist_bn35_init':
            print('Optional initialization inventory skipped: files not present.')
            continue
        raise
    r = pd.read_csv(rp, low_memory=False)
    e = pd.read_csv(ep, low_memory=False)
    r = r[r.cohort == 'original'].copy()
    e = e[e.cohort == 'original'].copy()
    for label, df, required in [('results', r, ['run_key', 'activation', 'seed', 'test_acc']), ('epochs', e, ['run_key', 'activation', 'seed', 'epoch', 'val_loss'])]:
        absent = set(required) - set(df.columns)
        if absent:
            raise ValueError(f'{exp} {label}: missing columns {sorted(absent)}')
        if 'experiment' in df and set(df.experiment.dropna()) != {exp}:
            raise ValueError(f'{exp}: unexpected experiment identifier in {label}.')
        df['seed'] = pd.to_numeric(df.seed, errors='raise').astype(int)
    if r.run_key.duplicated().any():
        raise ValueError(f'{exp}: duplicate result run_key.')
    if e.duplicated(['run_key', 'epoch']).any():
        raise ValueError(f'{exp}: duplicate run/epoch.')
    if set(e.run_key) != set(r.run_key):
        raise ValueError(f'{exp}: epoch/result run keys disagree.')
    for col in ['test_acc', 'test_loss', 'stopped_at']:
        if col in r:
            r[col] = pd.to_numeric(r[col], errors='raise')
    for col in ['epoch', 'val_loss', 'train_loss', 'val_acc', 'train_acc']:
        if col in e:
            e[col] = pd.to_numeric(e[col], errors='raise')
    if not np.equal(e.epoch, e.epoch.astype(int)).all():
        raise ValueError('Noninteger epoch.')
    e['epoch'] = e.epoch.astype(int)
    if np.isinf(r.test_acc).any():
        raise ValueError(f'{exp}: infinite test accuracy.')
    finite_acc = r.test_acc.dropna()
    if not finite_acc.between(0, 1).all():
        raise ValueError('Expected test accuracy proportions, not percentages.')
    group_cols = ['activation'] + (['initializer'] if exp == 'mnist_bn35_init' else [])
    for key, g in r.groupby(group_cols):
        if sorted(g.seed.tolist()) != SEEDS:
            raise ValueError(f'{exp}, {key}: expected seeds {SEEDS}; found {g.seed.tolist()}')
    if exp in EXPERIMENTS:
        expected = set(CONSTRUCTED + LITERATURE + (['ReLU'] if exp == 'cifar10_resnet32' else []))
        if set(r.activation) != expected:
            raise ValueError(f'{exp}: missing/extra functions: {expected.symmetric_difference(set(r.activation))}')
    for run, g in e.groupby('run_key'):
        if not np.array_equal(np.sort(g.epoch.to_numpy()), np.arange(1, int(g.epoch.max()) + 1)):
            raise ValueError(f'{exp}, {run}: noncontiguous epoch history.')
    if 'stopped_at' in r:
        stop = r.set_index('run_key').stopped_at
        last = e.groupby('run_key').epoch.max().reindex(stop.index)
        bad = stop.notna() & last.ne(stop)
        if bad.any():
            raise ValueError(f'{exp}: incomplete epoch export for {stop[bad].index.tolist()}')
    meta = e[['run_key', 'activation', 'seed']].drop_duplicates().merge(r[['run_key', 'activation', 'seed']], on='run_key', suffixes=('_epoch', '_result'), validate='one_to_one')
    if not (meta.activation_epoch.eq(meta.activation_result) & meta.seed_epoch.eq(meta.seed_result)).all():
        raise ValueError(f'{exp}: inconsistent activation/seed metadata.')
    results[exp] = r
    epochs[exp] = e.sort_values(['activation', 'seed', 'epoch'])
    audit_rows.append(dict(experiment=exp, functions=r.activation.nunique(), runs=len(r), epoch_rows=len(e), finite_test_runs=int(np.isfinite(r.test_acc).sum())))
audit = pd.DataFrame(audit_rows)
print(audit)
audit.to_csv(OUTPUT_DIR / 'input_inventory.csv', index=False)
print('All input checks passed. CSVs have not been modified.')
STAGES = ['loss_10pct', 'loss_20pct', 'loss_50pct', 'best_val_loss']
STAGE_LABELS = ['10%', '20%', '50%', 'Best']
OUTCOMES = {}
excluded_rows = []
for exp, (title, budget) in EXPERIMENTS.items():
    r = results[exp]
    e = epochs[exp]
    d = r[['run_key', 'activation', 'seed', 'test_acc']].copy()
    d['test_acc'] = d.test_acc.round(12)
    if 'test_loss' in r:
        d['test_loss'] = r.test_loss.to_numpy()
    for fraction in [0.1, 0.2, 0.5]:
        epoch = int(budget * fraction)
        d[f'loss_{int(100 * fraction)}pct'] = d.run_key.map(e.loc[e.epoch.eq(epoch)].set_index('run_key').val_loss)
    finite_e = e[np.isfinite(e.val_loss)]
    d['best_val_loss'] = d.run_key.map(finite_e.groupby('run_key').val_loss.min())
    OUTCOMES[exp] = d
    for name, g in d.groupby('activation'):
        if not np.isfinite(g.test_acc).all():
            excluded_rows.append(dict(experiment=exp, activation=name, finite_test_seeds=int(np.isfinite(g.test_acc).sum()), reason='Not all five test outcomes finite; no accuracy imputed'))
excluded = pd.DataFrame(excluded_rows, columns=['experiment', 'activation', 'finite_test_seeds', 'reason'])
excluded.to_csv(OUTPUT_DIR / 'excluded_functions.csv', index=False)
if len(excluded):
    print(excluded)

def endpoint_summary(exp, outcome, panel=PANEL):
    names = CONSTRUCTED if panel == 'constructed' else LITERATURE
    d = OUTCOMES[exp]
    if outcome not in d:
        return pd.DataFrame(columns=['mean', 'sd', 'n'])
    eligible = d.groupby('activation').test_acc.apply(lambda v: len(v) == len(SEEDS) and np.isfinite(v).all())
    mat = d[d.activation.isin(names)].pivot(index='activation', columns='seed', values=outcome).reindex(columns=SEEDS)
    mat = mat.loc[mat.index.intersection(eligible[eligible].index)]
    mat = mat[np.isfinite(mat).all(axis=1)]
    ans = pd.DataFrame({'mean': mat.mean(axis=1), 'sd': mat.std(axis=1, ddof=1), 'n': len(SEEDS)})
    if outcome == 'test_acc':
        ans['mean'] = ans['mean'].round(12)
    return ans

def rank_association(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return (np.nan, np.nan)
    return (float(spearmanr(x, y)[0]), float(kendalltau(x, y)[0]))
corr_rows = []
for exp in EXPERIMENTS:
    for outcome in STAGES + ['test_acc', 'test_loss']:
        sm = endpoint_summary(exp, outcome)
        if sm.empty:
            continue
        y = sm['mean'].to_numpy() * (1 if outcome == 'test_acc' else -1)
        for stat in predictors.columns:
            x = predictors.loc[sm.index, stat].to_numpy()
            rho, tau = rank_association(x, y)
            corr_rows.append(dict(experiment=exp, panel=PANEL, outcome=outcome, statistic=stat, n=len(sm), spearman=rho, kendall_tau_b=tau))
correlations = pd.DataFrame(corr_rows)
correlations.to_csv(OUTPUT_DIR / 'correlations_used.csv', index=False)
print(correlations[(correlations.statistic == RA) & (correlations.outcome == 'test_acc')][['experiment', 'n', 'spearman', 'kendall_tau_b']].reset_index(drop=True))
PROFILE_FUNCTIONS = ['Mish', 'Swish', 'BandHigh', 'BandLow']
PROFILE_STYLES = {'Mish': dict(color='#0072B2', linestyle='-'), 'Swish': dict(color='#555555', linestyle='--'), 'BandHigh': dict(color='#D55E00', linestyle='-.'), 'BandLow': dict(color='#009E73', linestyle=':')}
X_RANGE = (-3, 3)
S_RANGE = (0, 3)
FIG1_SIZE = (7.2, 2.8)
x = np.linspace(*X_RANGE, 601)
scales = np.unique(np.r_[np.linspace(*S_RANGE, 161), 0.25, 1.0, 3.0])
scales = scales[(scales >= S_RANGE[0]) & (scales <= S_RANGE[1])]
fig, axes = plt.subplots(1, 2, figsize=FIG1_SIZE, layout='constrained')
profile_rows = []
for i, name in enumerate(PROFILE_FUNCTIONS):
    style = PROFILE_STYLES.get(name, dict(color=f'C{i}', linestyle='-'))
    label = SHORT_NAMES.get(name, name)
    axes[0].plot(x, activation(name, x), label=label, **style)
    vals = np.array([gaussian_A(name, float(s)) for s in scales])
    axes[1].plot(scales, vals, label=label, **style)
    for s, a in zip(scales, vals):
        profile_rows.append(dict(activation=name, s=s, A=a))
    if name in predictors.index:
        computed = gaussian_A(name, 0.25)
        expected = predictors.loc[name, 'A(0.25)']
        if not np.isclose(computed, expected, atol=2e-08, rtol=0):
            raise ValueError(f'{name}: plotted A(.25) disagrees with embedded predictor.')
for ax in axes:
    ax.grid(alpha=0.18, linewidth=0.6)
axes[0].axhline(0, color='0.7', lw=0.6, zorder=0)
axes[0].axvline(0, color='0.7', lw=0.6, zorder=0)
axes[0].set(xlabel='$x$', ylabel='$f(x)$', xlim=X_RANGE)
axes[1].set(xlabel='Gaussian input scale $s$', ylabel='$A_f(s)$', xlim=S_RANGE)
for s in [0.25, 1.0]:
    axes[1].axvline(s, color='0.65', lw=0.8, ls='--', zorder=0)
if S_RANGE == (0, 3):
    axes[1].set_xticks([0, 0.25, 1, 2, 3])
    axes[1].set_xticklabels(['0', '.25', '1', '2', '3'])
panel_title(axes[0], 'a', 'Activation shapes')
panel_title(axes[1], 'b', 'Gaussian derivative averages')
axes[0].legend(frameon=False, loc='upper left')
pd.DataFrame(profile_rows).to_csv(OUTPUT_DIR / 'figure1_profile_values.csv', index=False)
finish_figure(fig, 'profiles')
FIG2_EXPERIMENTS = list(EXPERIMENTS)
FIG2_STATS = ['A(0.25)', 'B(0.25)', 'GammaBN(0.25)', RA, RB, 'B(1)']
FIG2_SIZE = (7.2, 7.4)
ncols = 3
nrows = math.ceil(len(FIG2_EXPERIMENTS) / ncols)
fig, axs = plt.subplots(nrows, ncols, figsize=FIG2_SIZE, sharey=True, squeeze=False)
for i, (ax, exp) in enumerate(zip(axs.flat, FIG2_EXPERIMENTS)):
    d = correlations[(correlations.experiment == exp) & correlations.outcome.isin(STAGES)]
    for stat in FIG2_STATS:
        y = d[d.statistic == stat].set_index('outcome').reindex(STAGES).spearman.to_numpy()
        style = STAT_STYLES.get(stat, dict(color=f'C{FIG2_STATS.index(stat)}', marker='o', linestyle='-'))
        ax.plot(range(4), y, markersize=4, linewidth=1.4, label=STAT_LABELS.get(stat, stat), **style)
    ns = sorted(d.n.unique())
    ax.text(0.96, 0.04, 'n=' + '/'.join(map(str, ns)), transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    ax.axhline(0, color='0.55', lw=0.7, zorder=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(axis='y', alpha=0.18, linewidth=0.6)
    panel_title(ax, chr(97 + i), EXPERIMENTS[exp][0])
    if i % ncols == 0:
        ax.set_ylabel('Spearman $r_S$ with $-L_{\\rm val}$')
for ax in list(axs.flat)[len(FIG2_EXPERIMENTS):]:
    ax.set_visible(False)
handles, labels = axs.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, columnspacing=1.8, handlelength=2.6)
fig.subplots_adjust(left=0.09, right=0.985, top=0.96, bottom=0.13, wspace=0.25, hspace=0.38)
correlations[correlations.experiment.isin(FIG2_EXPERIMENTS) & correlations.statistic.isin(FIG2_STATS) & correlations.outcome.isin(STAGES)].to_csv(OUTPUT_DIR / 'figure2_stage_correlations.csv', index=False)
finish_figure(fig, 'stage_correlations')
FIG3_SIZE = (7.2, 2.9)
FIG3_EXP = 'cifar10_resnet32'
FIG3_PREDICTOR = PRIMARY_PREDICTOR
FIG3_OUTCOMES = ['loss_20pct', 'best_val_loss', 'test_acc']
FIG3_TITLES = ['Early validation loss', 'Best validation loss', 'Test accuracy']
FIG3_LABEL_FUNCTIONS = ['BandHigh', 'BandLow', 'SetD1_LowA_HighGamma', 'SetD6_HighA_HighGamma']
LABEL_POSITIONS = {'loss_20pct': {'BandHigh': (0.77, 0.34), 'BandLow': (0.83, 0.025), 'SetD1_LowA_HighGamma': (0.46, 0.08), 'SetD6_HighA_HighGamma': (0.19, 0.08)}, 'best_val_loss': {'BandHigh': (0.79, 0.31), 'BandLow': (0.76, 0.08), 'SetD1_LowA_HighGamma': (0.41, 0.05), 'SetD6_HighA_HighGamma': (0.16, 0.1)}, 'test_acc': {'BandHigh': (0.84, 0.72), 'BandLow': (0.64, 0.62), 'SetD1_LowA_HighGamma': (0.42, 0.7), 'SetD6_HighA_HighGamma': (0.16, 0.73)}}
fig, axes = plt.subplots(1, 3, figsize=FIG3_SIZE, layout='constrained')
scatter_rows = []
for i, (ax, outcome, title) in enumerate(zip(axes, FIG3_OUTCOMES, FIG3_TITLES)):
    sm = endpoint_summary(FIG3_EXP, outcome)
    xs = predictors.loc[sm.index, FIG3_PREDICTOR]
    scale = 100 if outcome == 'test_acc' else 1
    ax.errorbar(xs, scale * sm['mean'], yerr=scale * sm.sd, fmt='o', ms=3.7, color='#0072B2', ecolor='#A8C8DB', elinewidth=0.8, capsize=1.5, markeredgecolor='white', markeredgewidth=0.35, zorder=3)
    orientation = 1 if outcome == 'test_acc' else -1
    rho, tau = rank_association(xs, orientation * sm['mean'])
    ax.text(0.04, 0.96, f'$r_S = {rho:.3f}$' + f'; n = {len(sm)}', transform=ax.transAxes, ha='left', va='top', fontsize=7.5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.8))
    for name in FIG3_LABEL_FUNCTIONS:
        if name not in sm.index:
            continue
        position = LABEL_POSITIONS.get(outcome, {}).get(name, (0.5, 0.5))
        ax.annotate(SHORT_NAMES.get(name, name), (xs.loc[name], scale * sm.loc[name, 'mean']), xytext=position, textcoords='axes fraction', fontsize=7, ha='center', va='center', arrowprops=dict(arrowstyle='-', lw=0.55, color='0.45'), bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.8))
    ax.margins(x=0.2, y=0.17)
    ax.set_xlabel(STAT_LABELS.get(FIG3_PREDICTOR, FIG3_PREDICTOR))
    ax.set_ylabel('Test accuracy (%)' if outcome == 'test_acc' else 'Validation cross-entropy')
    ax.grid(alpha=0.16, linewidth=0.6)
    panel_title(ax, chr(97 + i), title)
    for name in sm.index:
        scatter_rows.append(dict(activation=name, outcome=outcome, predictor=FIG3_PREDICTOR, x=xs.loc[name], mean=sm.loc[name, 'mean'], sd=sm.loc[name, 'sd'], n_seeds=sm.loc[name, 'n'], spearman_oriented=rho, kendall_tau_b=tau))
pd.DataFrame(scatter_rows).to_csv(OUTPUT_DIR / 'figure3_resnet_values.csv', index=False)
finish_figure(fig, 'resnet_endpoints')
CURVE_EXPERIMENTS = ['fashionmnist_bn35', 'cifar10_resnet32']
CURVE_PAIRS = [('Mish', 'Swish'), ('BandHigh', 'BandLow'), ('SetD1_LowA_HighGamma', 'SetD6_HighA_HighGamma')]
FIG4_SIZE = (7.2, 5.7)
CURVE_COLOURS = ['#0072B2', '#D55E00']
CURVE_LOG_Y = False
SHOW_SEED_TRACES = True
SHOW_SD_BANDS = True
fig, axes = plt.subplots(len(CURVE_EXPERIMENTS), len(CURVE_PAIRS), figsize=FIG4_SIZE, squeeze=False)
curve_rows = []
for row, exp in enumerate(CURVE_EXPERIMENTS):
    for col, pair in enumerate(CURVE_PAIRS):
        ax = axes[row, col]
        for name, colour in zip(pair, CURVE_COLOURS):
            d = epochs[exp][epochs[exp].activation.eq(name)]
            if set(d.seed) != set(SEEDS):
                raise ValueError(f'{exp}, {name}: incomplete curve seeds.')
            for metric, linestyle in [('val_loss', '-'), ('train_loss', '--')]:
                if metric not in d or not np.isfinite(d[metric]).any():
                    raise ValueError(f'{exp}: {metric} is not recorded; cannot draw it.')
                mat = d.pivot(index='epoch', columns='seed', values=metric).reindex(columns=SEEDS).sort_index()
                mat = mat.where(np.isfinite(mat))
                if SHOW_SEED_TRACES:
                    for seed in SEEDS:
                        ax.plot(mat.index, mat[seed], color=colour, ls=linestyle, lw=0.5, alpha=0.14, zorder=1)
                n = mat.notna().sum(axis=1)
                mu = mat.mean(axis=1).where(n.eq(len(SEEDS)))
                sd = mat.std(axis=1, ddof=1).where(n.eq(len(SEEDS)))
                ax.plot(mat.index, mu, color=colour, ls=linestyle, lw=1.6, zorder=3)
                if SHOW_SD_BANDS:
                    ax.fill_between(mat.index, mu - sd, mu + sd, color=colour, alpha=0.075, lw=0, zorder=2)
                for epoch in mat.index:
                    curve_rows.append(dict(experiment=exp, activation=name, metric=metric, epoch=epoch, n_recorded=int(n.loc[epoch]), mean_all5=mu.loc[epoch], sd_all5=sd.loc[epoch]))
        letters = chr(97 + row * len(CURVE_PAIRS) + col)
        names = ' / '.join((SHORT_NAMES.get(n, n) for n in pair))
        panel_title(ax, letters, names)
        ax.set_xlim(1, EXPERIMENTS[exp][1])
        ax.set_xlabel('Epoch')
        if col == 0:
            dataset = 'Fashion-MNIST BN35' if exp == 'fashionmnist_bn35' else 'CIFAR-10 ResNet-32'
            ax.set_ylabel(dataset + '\nCross-entropy')
        if CURVE_LOG_Y:
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15, linewidth=0.6)
        handles = [Line2D([0], [0], color=c, lw=2, label=SHORT_NAMES.get(n, n)) for n, c in zip(pair, CURVE_COLOURS)]
        ax.legend(handles=handles, frameon=False, loc='upper right', fontsize=7)
handles = [Line2D([0], [0], color='0.2', lw=1.6, ls='-', label='Validation'), Line2D([0], [0], color='0.2', lw=1.6, ls='--', label='Training')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), frameon=False, ncol=2, handlelength=3)
fig.subplots_adjust(left=0.1, right=0.985, top=0.95, bottom=0.12, wspace=0.3, hspace=0.4)
pd.DataFrame(curve_rows).to_csv(OUTPUT_DIR / 'figure4_curve_values.csv', index=False)
finish_figure(fig, 'learning_curves')
for stem in ['profiles', 'stage_correlations', 'resnet_endpoints', 'learning_curves']:
    path = OUTPUT_DIR / f'{stem}.pdf'
    if not path.is_file():
        raise FileNotFoundError(f'Run the figure cell for {stem}.')
    print(f'{path.name}: {path.stat().st_size / 1024:.1f} KiB')
    print('\\includegraphics[width=\\linewidth]{figures/' + stem + '.pdf}')
print('\nDone. PDFs, optional SVGs, and plotting-data CSVs are in:', OUTPUT_DIR.resolve())
