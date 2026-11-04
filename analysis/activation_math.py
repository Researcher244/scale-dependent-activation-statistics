"""NumPy activation values and analytic derivatives used for Gaussian descriptors."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit, ndtr, erf
ROOT = Path(__file__).resolve().parents[1]
params = json.loads((ROOT / 'data/reference/constructed_parameters.json').read_text())
constructed = list(params)
ADDITIONAL = {'Logish', 'Smish', 'Gish', 'S-Mish', 'CoLU'}

def original_fd(n, x):
    x = np.asarray(x)
    q = expit(x)
    sp = np.logaddexp(0.0, x)
    if n in constructed:
        p = params[n]
        if 'scale' in p:
            c = p['scale']
            return (c * x * q, c * (q + x * q * (1 - q)))
        beta = p['beta']
        b = p.get('b', 0)
        a = p.get('alpha', 0)
        g = p.get('gamma', p.get('eta', 0))
        q = expit(beta * x + b)
        e = np.exp(-x * x / 2)
        en = np.exp(-x * x / (2 * 0.15 ** 2))
        return (x * q + a * x * e + g * x * en, q + beta * x * q * (1 - q) + a * e * (1 - x * x) + g * en * (1 - x * x / 0.15 ** 2))
    if n == 'Mish':
        g = np.tanh(sp)
        gp = (1 - g * g) * q
    elif n == 'Swish':
        g = q
        gp = q * (1 - q)
    elif n == 'GeluAct':
        g = ndtr(x)
        gp = np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    elif n == 'SerfAct':
        g = erf(sp)
        gp = 2 / np.sqrt(np.pi) * np.exp(-sp * sp) * q
    elif n == 'TanhExpAct':
        v = np.exp(np.minimum(x, 20))
        g = np.tanh(v)
        gp = v * (1 - g * g)
    elif n == 'PhishAct':
        v = x * ndtr(x)
        d = ndtr(x) + x * np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
        g = np.tanh(v)
        gp = (1 - g * g) * d
    elif n == 'ExpExpishAct':
        v = np.exp(np.minimum(-x, 20))
        g = np.exp(-v)
        gp = v * g
    elif n == 'SquareSwishAct':
        g = q * q
        gp = 2 * q * q * (1 - q)
    return (x * g, g + x * gp)

def additional_fd(n, x):
    x = np.asarray(x)
    q = expit(x)
    qp = q * (1 - q)
    if n == 'Logish':
        g = np.log1p(q)
        gp = qp / (1 + q)
    elif n == 'Smish':
        g = np.tanh(np.log1p(q))
        gp = (1 - g * g) * qp / (1 + q)
    elif n == 'Gish':
        u = np.exp(np.minimum(x, 20))
        e = np.exp(-u)
        g = np.log1p(-np.expm1(-u))
        gp = u * e / (2 - e)
    elif n == 'S-Mish':
        g = 2 * q / (1 + q * q)
        gp = 2 * (1 - q * q) * qp / (1 + q * q) ** 2
    elif n == 'CoLU':
        xn = np.minimum(x, 0)
        u = np.exp(xn)
        e = np.exp(xn + u)
        ep = e * (1 + u)
        gn = e / (e - xn)
        gpn = (e - xn * ep) / (e - xn) ** 2
        xp = np.maximum(x, 0)
        u = np.exp(np.minimum(xp, 20))
        e = np.exp(-xp - u)
        v = xp * e
        g = np.where(x >= 0, 1 / (1 - v), gn)
        gp = np.where(x >= 0, e * (1 - xp * (1 + u)) / (1 - v) ** 2, gpn)
    return (x * g, g + x * gp)

def fd(name, x):
    return additional_fd(name, x) if name in ADDITIONAL else original_fd(name, x)
