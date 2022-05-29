import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from plif_node import NEURON
from torch.autograd import Variable


PRIMITIVES = {
  'skip_connect': lambda C_in, C_out, stride, snn_params: Identity(C_in, C_out, snn_params) if stride == 1 else FactorizedReduce(C_in, C_out, snn_params),
  'max_pool_k2' : lambda C_in, C_out, stride, snn_params: SpikingMaxPool2d(2, stride, snn_params),
  'avg_pool_k2' : lambda C_in, C_out, stride, snn_params: SpikingAvgPool2d(2, stride, snn_params),
  'SCB_k3': lambda C_in, C_out, stride, snn_params: SpikingConvBlock(C_in, C_out, 3, stride, snn_params),
  'SCB_k5': lambda C_in, C_out, stride, snn_params: SpikingConvBlock(C_in, C_out, 5, stride, snn_params),
  'SCB_k7': lambda C_in, C_out, stride, snn_params: SpikingConvBlock(C_in, C_out, 7, stride, snn_params),
  'SRB_k3': lambda C_in, C_out, stride, snn_params: SpikingResidualBlock(C_in, C_out, 3, stride, snn_params),
  'SRB_k5': lambda C_in, C_out, stride, snn_params: SpikingResidualBlock(C_in, C_out, 5, stride, snn_params),
  'SRB_k7': lambda C_in, C_out, stride, snn_params: SpikingResidualBlock(C_in, C_out, 7, stride, snn_params),
  'SIB_k3_e1': lambda C_in, C_out, stride, snn_params: SpikingInvertedBottleneck(C_in, C_out, 3, 1, stride, snn_params),
  'SIB_k3_e3': lambda C_in, C_out, stride, snn_params: SpikingInvertedBottleneck(C_in, C_out, 3, 3, stride, snn_params),
  'SIB_k3_e6': lambda C_in, C_out, stride, snn_params: SpikingInvertedBottleneck(C_in, C_out, 3, 6, stride, snn_params),
  'SIB_k5_e1': lambda C_in, C_out, stride, snn_params: SpikingInvertedBottleneck(C_in, C_out, 5, 1, stride, snn_params),
  'SIB_k5_e3': lambda C_in, C_out, stride, snn_params: SpikingInvertedBottleneck(C_in, C_out, 5, 3, stride, snn_params),
  'SIB_k5_e6': lambda C_in, C_out, stride, snn_params: SpikingInvertedBottleneck(C_in, C_out, 5, 6, stride, snn_params),
}

class Identity(nn.Module):

    def __init__(self, C_in, C_out, snn_params=None):
        super(Identity, self).__init__()
        self.identity = C_in == C_out
        if not self.identity:
            # SNN parameters
            if snn_params is not None:
                init_tau = snn_params['init_tau']
                v_threshold = snn_params['v_threshold']
                neuron = snn_params['neuron']

            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out)
            self.spike_neuron = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

    def forward(self, x):
        if self.identity:
            return x, 0
        else:
            spikes = 0
            out = self.bn(self.conv(x))
            if self.spike_neuron:
                out = self.spike_neuron(out)
                spikes += out.sum().item()
            else: # for ANN
                out = F.relu(out)
            return out, spikes

class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, snn_params=None):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0

        # SNN parameters
        if snn_params is not None:
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']

        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self.spike_neuron = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

    def forward(self, x): 
      spikes = 0
      out = self.bn(torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1))
          
      if self.spike_neuron:
          out = self.spike_neuron(out)
          spikes += out.sum().item()
      else:
          out = F.relu(out)
      return out, spikes

""" 2D max pooling
"""
class SpikingMaxPool2d(nn.Module):

    def __init__(self, kernel_size, stride, snn_params=None):
        super(SpikingMaxPool2d, self).__init__()

        # SNN parameters
        if snn_params is not None:
            self.neuron = snn_params['neuron']

        self.maxpool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        spikes = 0
        out = self.maxpool(x)
        if not(self.neuron == 'ANN'):
            spikes += out.sum().item()
        return out, spikes

""" 2D avg pooling
"""
class SpikingAvgPool2d(nn.Module):

    def __init__(self, kernel_size, stride, snn_params=None):
        super(SpikingAvgPool2d, self).__init__()

        # SNN parameters
        if snn_params is not None:
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            self.neuron = snn_params['neuron']

        self.avgpool = nn.AvgPool2d(kernel_size, stride)
        self.spike_neuron = None if self.neuron == 'ANN' else NEURON[self.neuron](init_tau, v_threshold)

    def forward(self, x):
        spikes = 0
        out = self.avgpool(x)
        if not(self.neuron == 'ANN'):
            out = self.spike_neuron(out)
            spikes += out.sum().item()
        return out, spikes



""" Spiking ConvBlock
    - two conv
"""
class SpikingConvBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride = 1, snn_params = None):
        super(SpikingConvBlock, self).__init__()

        self.stride = stride
        # SNN parameters
        if snn_params is not None:
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']

        # first conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size//2), groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spike_neuron_1 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        # second conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=(kernel_size//2), groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spike_neuron_2 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

    def forward(self, x):
        spikes = 0
        out = self.bn1(self.conv1(x))
        if self.spike_neuron_1:
            out = self.spike_neuron_1(out)
            tmp_spikes = out.sum().item()
            spikes += tmp_spikes
        else:
            out = F.relu(out)

        out = self.bn2(self.conv2(out))
        if self.spike_neuron_2:
            out = self.spike_neuron_2(out)
            tmp_spikes = out.sum().item()
            spikes += tmp_spikes
        else:
            out = F.relu(out)
            
        return out, spikes

""" Spiking ResidualBlock
    - two conv and skip connection
"""
class SpikingResidualBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride = 1, snn_params = None):
        super(SpikingResidualBlock, self).__init__()

        self.stride = stride
        # SNN parameters
        if snn_params is not None:
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']

        # first conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size//2), groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spike_neuron_1 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        # second conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=(kernel_size//2), groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spike_neuron_2 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        # skip
        self.use_res_connect = stride == 1 and inplanes == planes
        if not self.use_res_connect:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        spikes = 0
        out = self.bn1(self.conv1(x))
        if self.spike_neuron_1:
            out = self.spike_neuron_1(out)
            tmp_spikes = out.sum().item()
            spikes += tmp_spikes
        else:
            out = F.relu(out)

        out = self.bn2(self.conv2(out))
        if self.use_res_connect:
            out = out + x
        else:
            out = out + self.downsample(x)
        if self.spike_neuron_2:
            out = self.spike_neuron_2(out)
            tmp_spikes = out.sum().item()
            spikes += tmp_spikes
        else:
            out = F.relu(out)
            
        return out, spikes

""" Spiking InvertedBottleneck
"""
class SpikingInvertedBottleneck(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, expansion, stride, snn_params=None):
        super(SpikingInvertedBottleneck, self).__init__()

        self.stride = stride
        # SNN parameters
        if snn_params is not None:
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']

        # first conv (point-wise)
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spike_neuron_1 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        # second conv (depth-wise)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size//2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spike_neuron_2 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        # third conv (point-wise)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.spike_neuron_3 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        self.use_res_connect = self.stride == 1 and in_planes == out_planes
        self.downsample = None
        if stride == 1 and in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        # when stride == 2, there is no downsample connection (MobileNetV2 paper, Figure 4-(d))

    def forward(self, x):
        spikes = 0
        out = self.bn1(self.conv1(x))
        if self.spike_neuron_1:
            out = self.spike_neuron_1(out)
            tmp_spikes = out.sum().item()
            spikes += tmp_spikes
        else:
            out = F.relu(out)

        out = self.bn2(self.conv2(out))
        if self.spike_neuron_2:
            out = self.spike_neuron_2(out)
            tmp_spikes = out.sum().item()
            spikes += tmp_spikes
        else:
            out = F.relu(out)

        out = self.bn3(self.conv3(out))
        if self.use_res_connect:
            out = out + x
        else: 
            if self.downsample is not None:
                out = out + self.downsample(x)
        if self.spike_neuron_3:
            out = self.spike_neuron_3(out)
            tmp_spikes = out.sum().item()
            spikes += tmp_spikes
        else:
            out = F.relu(out)

        return out, spikes
