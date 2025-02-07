"""
Radial basis function embedding from DimeNet:
Gasteiger, Johannes, Janek Groß, and Stephan Günnemann.
"Directional message passing for molecular graphs."
arXiv preprint arXiv:2003.03123 (2020).

https://github.com/gasteigerjo/dimenet/tree/master/dimenet/model/layers
"""
import math
import torch
from torch import nn


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    Eq. (8) in the paper
    """
    def __init__(self, p=6):
        super(Envelope, self).__init__()
        self.p = p
        self.const1 = -(self.p + 1) * (self.p + 2) / 2
        self.const2 = self.p * (self.p + 2)
        self.const3 = -self.p * (self.p + 1) / 2

    def forward(self, d):
        return 1 + self.const1 * d**self.p + self.const2 * d**(self.p + 1) + \
               self.const3 * d**(self.p + 2)


class RadialBesselBasis(nn.Module):
    def __init__(self, cutoff, num_radial=16, envelope_p=6):
        super(RadialBesselBasis, self).__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_p)

        self.const = math.sqrt(2 / self.cutoff)

        self.freq = torch.arange(1, self.num_radial + 1) * math.pi / self.cutoff
        self.freq = self.freq.unsqueeze(0)

    def forward(self, d):
        """
        Eq. (7) of the paper
        :param d: (n,)
        :return: distance embedding (n, num_radial)
        """
        d = d.view(-1, 1)  # for broadcasting
        freq = self.freq.to(d.device)
        return self.const * torch.sin(freq * d) / d * self.envelope(d)
