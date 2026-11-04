"""Reference PyTorch definitions of the 35 fixed activations.
Training scripts keep their original local definitions. ReLU is ResNet-only.
"""
import torch
from torch import nn
from torch.nn import functional as F

def mish(x):
    return x * torch.tanh(F.softplus(x))

def swish(x):
    return x * torch.sigmoid(x)

class MishAct(nn.Module):

    def forward(self, x):
        return mish(x)

    def get_param(self):
        return {}

class SwishAct(nn.Module):

    def forward(self, x):
        return swish(x)

    def get_param(self):
        return {}

def gelu(x):
    return F.gelu(x)

class GeluAct(nn.Module):

    def forward(self, x):
        return gelu(x)

    def get_param(self):
        return {}

class FixC1MatchSlopeAndRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 1.0056290824
        self.alpha = 0.2481362485
        self.b = -0.6108569031

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixC1FixRhoMatchSlope(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 1.0655913532
        self.alpha = 0.2647998861
        self.b = -0.6847583639

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixC1FixSlopeMatchRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.9579116739
        self.alpha = 0.0
        self.b = 0.0

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

    def get_param(self):
        return {}

class FixRhoMatchC1AndSlope(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 2.5097657107
        self.alpha = -0.3622986508
        self.b = 3.239628968

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixRhoFixSlopeMatchC1(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 3.6454545499
        self.alpha = -0.4959679749
        self.b = 5.5094463356

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixSlopeMatchC1AndRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 3.2799942834
        self.alpha = -0.4929381821
        self.b = 4.9459659006

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixSlopeandC1HighRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.5
        self.alpha = 0.0
        self.b = 0.0

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class FixSlopeandC1LowRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 2.0
        self.alpha = 0.0
        self.b = 0.0

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class ScaledSwish(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = 1.4

    def forward(self, x):
        return self.scale * x * torch.sigmoid(x)

    def get_param(self):
        return {}

class HighBandSwish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = -1.0559
        self.b = -0.5061
        self.alpha = 0.1947
        self.gamma = -0.0708

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class LowBandSwish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = -1.0127
        self.b = 0.4625
        self.alpha = -0.1843
        self.gamma = 0.0707

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class BandHigh(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 1.20680028
        self.b = 0.04461946
        self.alpha = 0.16568067
        self.gamma = -0.07683369

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class BandLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 3.09266783
        self.b = 4.56834656
        self.alpha = -0.47355169
        self.gamma = 0.08382025

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class GentleTiltDown09(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 1.3212061309
        self.b = 0.6706517594
        self.alpha = -0.0277762821
        self.gamma = -0.1265

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.gamma * narrow

    def get_param(self):
        return {}

class MatchAllToMish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 2.0996330505
        self.alpha = -0.3271993629
        self.b = 2.5444438979

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * x * torch.exp(-0.5 * x ** 2)

    def get_param(self):
        return {}

class ExpExpishAct(nn.Module):

    def forward(self, x):
        return x * torch.exp(-torch.exp(-x))

    def get_param(self):
        return {}

class SerfAct(nn.Module):

    def forward(self, x):
        return x * torch.erf(F.softplus(x))

    def get_param(self):
        return {}

class TanhExpAct(nn.Module):

    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

    def get_param(self):
        return {}

class PhishAct(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.gelu(x))

    def get_param(self):
        return {}

class SquareSwishAct(nn.Module):

    def forward(self, x):
        s = torch.sigmoid(x)
        return x * s * s

    def get_param(self):
        return {}

class SetD1_LowA_HighGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.4949478828
        self.b = -0.413756283
        self.alpha = 0.0388663503
        self.eta = 0.0027885451

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD2_LowA_LowGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.42081386
        self.b = 0.48359119
        self.alpha = -0.18791026
        self.eta = -0.00805122

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD3_MidA_LowGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.3093398248
        self.b = 1.0044482442
        self.alpha = -0.223628617
        self.eta = -0.0137558189

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD4_MidA_HighGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.916394602
        self.b = 0.0436458135
        self.alpha = 0.0264917301
        self.eta = 0.0022294333

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD5_HighA_LowGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.25
        self.b = 1.6646126199
        self.alpha = -0.2443422477
        self.eta = -0.015827546

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class SetD6_HighA_HighGamma(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = 0.3558473158
        self.b = 1.1158205221
        self.alpha = -0.1377719114
        self.eta = -0.009294866

    def forward(self, x):
        wide = x * torch.exp(-0.5 * x ** 2)
        narrow = x * torch.exp(-x ** 2 / (2 * 0.15 ** 2))
        return x * torch.sigmoid(self.beta * x + self.b) + self.alpha * wide + self.eta * narrow

    def get_param(self):
        return {}

class ScaledSwishhalf(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = 0.5

    def forward(self, x):
        return self.scale * x * torch.sigmoid(x)

    def get_param(self):
        return {}

class LogishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        return (z * torch.log1p(torch.sigmoid(z))).to(x.dtype)

    def get_param(self):
        return {}

class SmishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        return (z * torch.tanh(torch.log1p(torch.sigmoid(z)))).to(x.dtype)

    def get_param(self):
        return {}

class GishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        inner = torch.exp(torch.clamp(z, max=20.0))
        return (z * torch.log1p(-torch.expm1(-inner))).to(x.dtype)

    def get_param(self):
        return {}

class SMishAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        q = torch.sigmoid(z)
        return (z * (2.0 * q / (1.0 + q.square()))).to(x.dtype)

    def get_param(self):
        return {}

class CoLUAct(nn.Module):

    def forward(self, x):
        z = _new_act_work(x)
        neg_x = torch.where(z < 0, z, torch.zeros_like(z))
        e = torch.exp(neg_x + torch.exp(neg_x))
        negative = neg_x * e / (e - neg_x)
        pos_x = torch.where(z >= 0, z, torch.zeros_like(z))
        t = pos_x * torch.exp(-pos_x - torch.exp(torch.clamp(pos_x, max=20.0)))
        positive = pos_x / (1.0 - t)
        return torch.where(z >= 0, positive, negative).to(x.dtype)

    def get_param(self):
        return {}
ACTIVATIONS = {'Logish': LogishAct, 'Smish': SmishAct, 'Gish': GishAct, 'S-Mish': SMishAct, 'CoLU': CoLUAct, 'Mish': MishAct, 'Swish': SwishAct, 'FixC1MatchSlopeAndRho': FixC1MatchSlopeAndRho, 'FixC1FixRhoMatchSlope': FixC1FixRhoMatchSlope, 'FixC1FixSlopeMatchRho': FixC1FixSlopeMatchRho, 'FixRhoMatchC1AndSlope': FixRhoMatchC1AndSlope, 'FixRhoFixSlopeMatchC1': FixRhoFixSlopeMatchC1, 'FixSlopeMatchC1AndRho': FixSlopeMatchC1AndRho, 'FixSlopeandC1HighRho': FixSlopeandC1HighRho, 'FixSlopeandC1LowRho': FixSlopeandC1LowRho, 'MatchAllToMish': MatchAllToMish, 'BandHigh': BandHigh, 'BandLow': BandLow, 'GentleTiltDown09': GentleTiltDown09, 'ScaledSwish': ScaledSwish, 'HighBandSwish': HighBandSwish, 'LowBandSwish': LowBandSwish, 'GeluAct': GeluAct, 'ExpExpishAct': ExpExpishAct, 'SerfAct': SerfAct, 'TanhExpAct': TanhExpAct, 'PhishAct': PhishAct, 'SquareSwishAct': SquareSwishAct, 'SetD5_HighA_LowGamma': SetD5_HighA_LowGamma, 'ScaledSwishhalf': ScaledSwishhalf, 'SetD1_LowA_HighGamma': SetD1_LowA_HighGamma, 'SetD2_LowA_LowGamma': SetD2_LowA_LowGamma, 'SetD3_MidA_LowGamma': SetD3_MidA_LowGamma, 'SetD4_MidA_HighGamma': SetD4_MidA_HighGamma, 'SetD6_HighA_HighGamma': SetD6_HighA_HighGamma}

RESNET_EXTRA_REFERENCE = {'ReLU': nn.ReLU}
