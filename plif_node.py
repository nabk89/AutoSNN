# reference: 
# https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, accelerating
from spikingjelly.clock_driven.neuron import BaseNode, IFNode, LIFNode
import math

class PLIFNode(BaseNode):
    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=surrogate.ATan(), monitor_state=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        init_w = - math.log(init_tau - 1)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            # self.v += dv - self.v * self.w.sigmoid()
            self.v += (dv - self.v) * self.w.sigmoid()
        else:
            # self.v += dv - (self.v - self.v_reset) * self.w.sigmoid()
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        return self.spiking()

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'

NEURON = {
  'IF': lambda tau, v_threshold: IFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan(learnable=False), detach_reset=True),
  'LIF': lambda tau, v_threshold: LIFNode(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate.ATan(learnable=False), detach_reset=True),
  'PLIF': lambda tau, v_threshold: PLIFNode(init_tau=tau, v_threshold=v_threshold, surrogate_function=surrogate.ATan(learnable=False), detach_reset=True),
}
