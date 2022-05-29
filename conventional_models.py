import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.clock_driven import layer
from blocks import PRIMITIVES
from plif_node import NEURON

import logging

""" ref: Direct training for spiking neural networks: Faster, larger, better
"""
class CIFARNet_Wu(nn.Module):
    def __init__(self, num_class, snn_params, channels=128):
        super(CIFARNet_Wu, self).__init__()
    
        # SNN parameters
        self.snn_params = snn_params
        if snn_params is not None:
            self.T = snn_params['T']
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']
            if neuron == 'ANN':
                self.T = 1

        # build network
        img_channel=3
        self.conv_1 = nn.Sequential(
            nn.Conv2d(img_channel, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spike_neuron_1 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels*2),
        )
        self.spike_neuron_2 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.avg_pool_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            NEURON[neuron](init_tau, v_threshold),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels*4),
        )
        self.spike_neuron_3 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.avg_pool_3 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            NEURON[neuron](init_tau, v_threshold),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels*8),
        )
        self.spike_neuron_4 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        self.conv_5 = nn.Sequential(
            nn.Conv2d(channels*8, channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels*4),
        )
        self.spike_neuron_5 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels*4 * 8 * 8, channels * 8, bias=False),
        )
        self.spike_neuron_fc_1 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        self.fc_2 = nn.Sequential(
            nn.Linear(channels * 8, channels * 4, bias=False),
        )
        self.spike_neuron_fc_2 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)

        self.fc_3 = nn.Sequential(
            nn.Linear(channels * 4, num_class*10, bias=False),
        )
        self.spike_neuron_fc_3 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        num_of_spikes = 0
        out_spikes_counter = None

        x = self.conv_1(x)
        for t in range(self.T):
            out = self.spike_neuron_1(x)
            num_of_spikes += out.sum().item()

            out = self.spike_neuron_2(self.conv_2(out))
            num_of_spikes += out.sum().item()
            out = self.avg_pool_2(out)
            num_of_spikes += out.sum().item()
            
            out = self.spike_neuron_3(self.conv_3(out))
            num_of_spikes += out.sum().item()
            out = self.avg_pool_3(out)
            num_of_spikes += out.sum().item()

            out = self.spike_neuron_4(self.conv_4(out))
            num_of_spikes += out.sum().item()
            out = self.spike_neuron_5(self.conv_5(out))
            num_of_spikes += out.sum().item()

            out = self.fc_1(out)
            out = self.spike_neuron_fc_1(out)
            num_of_spikes += out.sum().item()
            out = self.fc_2(out)
            out = self.spike_neuron_fc_2(out)
            num_of_spikes += out.sum().item()
            out = self.fc_3(out)
            out = self.spike_neuron_fc_3(out)
            num_of_spikes += out.sum().item()

            if out_spikes_counter is None:
                out_spikes_counter = self.boost(out.unsqueeze(1)).squeeze(1)
            else:
                out_spikes_counter += self.boost(out.unsqueeze(1)).squeeze(1)

        return out_spikes_counter, num_of_spikes / x.shape[0]

def Spiking_CIFARNet_Wu(num_class, snn_params, init_channels=128):
    return CIFARNet_Wu(num_class, snn_params, init_channels)


""" ref: Incorporating learnable membrane time constant to enhance learning of spiking neural networks
"""
class CIFARNet_Fang(nn.Module):
    def __init__(self, num_class, snn_params, channels=256):
        super(CIFARNet_Fang, self).__init__()
    
        # SNN parameters
        self.snn_params = snn_params
        if snn_params is not None:
            self.T = snn_params['T']
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']
            if neuron == 'ANN':
                self.T = 1

        # build network
        img_channel=3
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(img_channel, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spike_neuron_1_1 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spike_neuron_1_2 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.conv_1_3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spike_neuron_1_3 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.max_pool_1 = nn.MaxPool2d(2, 2)

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spike_neuron_2_1 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spike_neuron_2_2 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.conv_2_3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spike_neuron_2_3 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.max_pool_2 = nn.MaxPool2d(2, 2)

        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, (channels//2) * 4 * 4, bias=False),
        )
        self.spike_neuron_fc_1 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.fc_2 = nn.Sequential(
            nn.Linear( (channels//2) * 4 * 4, num_class*10, bias=False),
        )
        self.spike_neuron_fc_2 = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        num_of_spikes = 0
        out_spikes_counter = None

        x = self.conv_1_1(x)
        for t in range(self.T):
            out = self.spike_neuron_1_1(x)
            num_of_spikes += out.sum().item()
            out = self.conv_1_2(out)
            out = self.spike_neuron_1_2(out)
            num_of_spikes += out.sum().item()
            out = self.conv_1_3(out)
            out = self.spike_neuron_1_3(out)
            num_of_spikes += out.sum().item()
            out = self.max_pool_1(out)
            num_of_spikes += out.sum().item()

            out = self.conv_2_1(out)
            out = self.spike_neuron_2_1(out)
            num_of_spikes += out.sum().item()
            out = self.conv_2_2(out)
            out = self.spike_neuron_2_2(out)
            num_of_spikes += out.sum().item()
            out = self.conv_2_3(out)
            out = self.spike_neuron_2_3(out)
            num_of_spikes += out.sum().item()
            out = self.max_pool_2(out)
            num_of_spikes += out.sum().item()

            out = self.fc_1(out)
            out = self.spike_neuron_fc_1(out)
            num_of_spikes += out.sum().item()
            out = self.fc_2(out)
            out = self.spike_neuron_fc_2(out)
            num_of_spikes += out.sum().item()

            if out_spikes_counter is None:
                out_spikes_counter = self.boost(out.unsqueeze(1)).squeeze(1)
            else:
                out_spikes_counter += self.boost(out.unsqueeze(1)).squeeze(1)

        return out_spikes_counter, num_of_spikes / x.shape[0]

def Spiking_CIFARNet_Fang(num_class, snn_params, init_channels=256):
    return CIFARNet_Fang(num_class, snn_params, init_channels)


""" ref: Incorporating learnable membrane time constant to enhance learning of spiking neural networks
"""
class DVS_CIFARNet_Fang(nn.Module):
    def __init__(self, num_class, num_layers, snn_params, channels=128):
        super(DVS_CIFARNet_Fang, self).__init__()
    
        assert(num_layers == 4 or num_layers == 5)
        # 4 = CIFAR10DVS, 5 = DVS128Gesture

        # SNN parameters
        self.snn_params = snn_params
        if snn_params is not None:
            self.T = snn_params['T']
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']

        # build network
        img_channel=2
        img_size = 128
        self.conv_1 = nn.Sequential(
            nn.Conv2d(img_channel, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            NEURON[neuron](init_tau, v_threshold)
        )
        self.max_pool_1 = nn.MaxPool2d(2, 2)
        img_size = img_size // 2

        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            NEURON[neuron](init_tau, v_threshold)
        )
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        img_size = img_size // 2

        self.conv_3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            NEURON[neuron](init_tau, v_threshold)
        )
        self.max_pool_3 = nn.MaxPool2d(2, 2)
        img_size = img_size // 2

        self.conv_4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            NEURON[neuron](init_tau, v_threshold)
        )
        self.max_pool_4 = nn.MaxPool2d(2, 2)
        img_size = img_size // 2

        if num_layers == 5:
            self.conv_5 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                NEURON[neuron](init_tau, v_threshold)
            )
            self.max_pool_5 = nn.MaxPool2d(2, 2)
            img_size = img_size // 2
        else:
            self.conv_5 = None

        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * img_size * img_size, channels * 4 * 4 // 4, bias=False),
            NEURON[neuron](init_tau, v_threshold)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear( channels * 4 * 4 // 4, num_class*10, bias=False),
            NEURON[neuron](init_tau, v_threshold)
        )
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        num_of_spikes = 0
        out_spikes_counter = None
        x = x.permute(1, 0, 2, 3, 4) # [T, N, 2, *, *]

        for t in range(x.shape[0]):
            out = self.conv_1(x[t])
            num_of_spikes += out.sum().item()
            out = self.max_pool_1(out)
            num_of_spikes += out.sum().item()

            out = self.conv_2(out)
            num_of_spikes += out.sum().item()
            out = self.max_pool_2(out)
            num_of_spikes += out.sum().item()

            out = self.conv_3(out)
            num_of_spikes += out.sum().item()
            out = self.max_pool_3(out)
            num_of_spikes += out.sum().item()

            out = self.conv_4(out)
            num_of_spikes += out.sum().item()
            out = self.max_pool_4(out)
            num_of_spikes += out.sum().item()

            if self.conv_5 is not None:
                out = self.conv_5(out)
                num_of_spikes += out.sum().item()
                out = self.max_pool_5(out)
                num_of_spikes += out.sum().item()

            out = self.fc_1(out)
            num_of_spikes += out.sum().item()
            out = self.fc_2(out)
            num_of_spikes += out.sum().item()

            if out_spikes_counter is None:
                out_spikes_counter = self.boost(out.unsqueeze(1)).squeeze(1)
            else:
                out_spikes_counter += self.boost(out.unsqueeze(1)).squeeze(1)

        return out_spikes_counter, num_of_spikes / x.shape[1]

def Spiking_DVS_CIFARNet_Fang(num_class, num_layers, snn_params, init_channels):
    return DVS_CIFARNet_Fang(num_class, num_layers, snn_params, init_channels)


""" ref: 
Enabling spike-based backpropagation for training deep neural network architectures
Going Deeper With Directly-Trained Larger Spiking Neural Networks
"""
class ResNet(nn.Module):
    def __init__(self, num_class, snn_params, model_spec):
        super(ResNet, self).__init__()

        # SNN parameters
        self.snn_params = snn_params
        if snn_params is not None:
            self.T = snn_params['T']
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']

        # build network
        img_size = 32
        C_stem   = model_spec['C_stem']
        channels = model_spec['channels']
        C_last = model_spec['C_last']
        strides  = model_spec['strides']
        use_downsample_avg = model_spec['use_downsample_avg'] 
        last_avg_pool = model_spec['last_avg_pool'] 

        # stem conv layer
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, C_stem, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )
        self.spike_neuron_stem = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        logging.info(f'stem conv 3x3\t{img_size} x {img_size} x 3\t-->\t{img_size} x {img_size} x {C_stem}')
        
        # (enabling spike-based learning)
        if use_downsample_avg:
            self.stem_avgpool = nn.Sequential(
                nn.AvgPool2d(2,2),
                NEURON[neuron](init_tau, v_threshold),
            ) 
            logging.info(f'stem avg 2x2\t{img_size} x {img_size} x {C_stem}\t-->\t{img_size//2} x {img_size//2} x {C_stem}')
            img_size = img_size//2
        else:
            self.stem_avgpool = None

        logging.info(f'------------ residual block layers ------------')

        # block layers
        self.layers = nn.ModuleList()
        C_in = C_stem
        for C_out, stride in zip(channels, strides):
            self.layers.append(PRIMITIVES['SRB_k3'](C_in, C_out, stride, self.snn_params))
            if stride == 1:
                logging.info(f'SRB_k3\t{img_size} x {img_size} x {C_in}\t-->\t{img_size} x {img_size} x {C_out}')
            elif stride == 2:
                logging.info(f'SRB_k3\t{img_size} x {img_size} x {C_in}\t-->\t{img_size//2} x {img_size//2} x {C_out}')
                img_size = img_size // 2
            C_in = C_out
        logging.info(f'-----------------------------------------------')
                
        if last_avg_pool == '2x2':
            # (going deeper)
            self.last_avgpool = nn.Sequential(
                nn.AvgPool2d(2,2),
                NEURON[neuron](init_tau, v_threshold),
            ) 
            logging.info(f'avgpool /2\t{img_size} x {img_size} x {C_in}\t-->\t{img_size//2} x {img_size//2} x {C_in}')
            img_size = img_size // 2
        elif last_avg_pool == 'GAP':
            self.last_avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                NEURON[neuron](init_tau, v_threshold),
            )
            logging.info(f'GAP\t{img_size} x {img_size} x {C_in}\t-->\t1 x 1 x {C_in}')
            img_size = 1
        else:
            self.last_avgpool = None

        # FC layer
        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C_in*img_size*img_size, C_last, bias=False),
            NEURON[neuron](init_tau, v_threshold),
        )
        logging.info(f'FC1     \t{C_in*img_size*img_size}\t-->\t{C_last}')

        self.fc_2 = nn.Sequential(
            nn.Linear(C_last, num_class*10, bias=False), ## before boost
            NEURON[neuron](init_tau, v_threshold),
        )
        logging.info(f'FC2     \t{C_last}\t-->\t{num_class*10}')
        self.boost = nn.AvgPool1d(10, 10)
        logging.info(f'boost   \t{num_class*10}\t-->\t{num_class}')
            
    def forward(self, x):
        num_of_spikes = 0
        out_spikes_counter = None

        x = self.conv_stem(x)
        for t in range(1, self.T):
            out = self.spike_neuron_stem(x)
            num_of_spikes += out.sum().item()
            if self.stem_avgpool is not None:
                out = self.stem_avgpool(out)
                num_of_spikes += out.sum().item()
            for layer in self.layers:
                out, spikes = layer(out)
                num_of_spikes += spikes
            if self.last_avgpool is not None:
                out = self.last_avgpool(out)
                num_of_spikes += out.sum().item()
            out = self.fc_1(out)
            num_of_spikes += out.sum().item()
            out = self.fc_2(out)
            num_of_spikes += out.sum().item()
            if out_spikes_counter is None:
                out_spikes_counter = self.boost(out.unsqueeze(1)).squeeze(1)
            else:
                out_spikes_counter += self.boost(out.unsqueeze(1)).squeeze(1)

        return out_spikes_counter, num_of_spikes / x.shape[0]

""" ref: Going Deeper With Directly-Trained Larger Spiking Neural Networks
"""
def Spiking_ResNet19_Zheng(num_class, snn_params, init_channels=128):
    c = init_channels
    model_spec = {
        'C_stem': c,
        'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
        'C_last': c*2, ## FC1,
        'strides': [1, 1, 1, 2, 1, 1, 2, 1],
        'use_downsample_avg': False,
        'last_avg_pool': '2x2',
    }
    return ResNet(num_class, snn_params, model_spec)

def Spiking_ResNet19_Zheng_no_GAP(num_class, snn_params, init_channels=128):
    c = init_channels
    model_spec = {
        'C_stem': c,
        'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
        'C_last': c*2, ## FC1,
        'strides': [1, 1, 1, 2, 1, 1, 2, 1],
        'use_downsample_avg': False,
        'last_avg_pool': None,
    }
    return ResNet(num_class, snn_params, model_spec)

def Spiking_ResNet19_Zheng_GAP(num_class, snn_params, init_channels=128):
    c = init_channels
    model_spec = {
        'C_stem': c,
        'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
        'C_last': c*2, ## FC1,
        'strides': [1, 1, 1, 2, 1, 1, 2, 1],
        'use_downsample_avg': False,
        'last_avg_pool': 'GAP',
    }
    return ResNet(num_class, snn_params, model_spec)



""" ref: Enabling spike-based backpropagation for training deep neural network architectures
"""
def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=64):
    c = init_channels
    model_spec = {
        'C_stem': c,
        'channels': [c*2, c*4, c*8, c*8],
        'C_last': c*16, ## FC1,
        'strides': [1, 2, 1, 2],
        'use_downsample_avg': True,
        'last_avg_pool': None,
    }
    return ResNet(num_class, snn_params, model_spec)


""" ref: Going Deeper With Directly-Trained Larger Spiking Neural Networks
"""
class DVS_ResNet(nn.Module):
    def __init__(self, num_class, snn_params, model_spec):
        super(DVS_ResNet, self).__init__()

        # SNN parameters
        self.snn_params = snn_params
        if snn_params is not None:
            self.T = snn_params['T']
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']

        # build network
        img_size = 128
        C_stem   = model_spec['C_stem']
        channels = model_spec['channels']
        C_last = model_spec['C_last']
        strides  = model_spec['strides']

        # stem conv layer
        self.conv_stem = nn.Sequential(
            nn.Conv2d(2, C_stem, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )
        self.spike_neuron_stem = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        logging.info(f'stem conv 3x3\t{img_size} x {img_size} x 2\t-->\t{img_size} x {img_size} x {C_stem}')
       
 
        # additional layers for DVS data (same composition in ours)
        logging.info(f'------- additional layers for DVS data -------')
        self.DVS_max_pool_1 = PRIMITIVES['max_pool_k2'](C_stem, C_stem, stride=2, snn_params=self.snn_params)
        logging.info(f'max_pool_k2\t{img_size} x {img_size} x {C_stem}\t-->\t{img_size // 2} x {img_size // 2} x {C_stem}')
        img_size = img_size // 2

        self.DVS_conv_1 = nn.Sequential(
            nn.Conv2d(C_stem, C_stem, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_stem),
            NEURON[neuron](init_tau, v_threshold),
        )
        logging.info(f'conv_k3   \t{img_size} x {img_size} x {C_stem}\t-->\t{img_size} x {img_size} x {C_stem}')
        self.DVS_max_pool_2 = PRIMITIVES['max_pool_k2'](C_stem, C_stem, stride=2, snn_params=self.snn_params)
        logging.info(f'max_pool_k2\t{img_size} x {img_size} x {C_stem}\t-->\t{img_size // 2} x {img_size // 2} x {C_stem}')
        img_size = img_size // 2

        
        # block layers (main blocks)
        logging.info(f'------------ residual block layers ------------')
        self.layers = nn.ModuleList()
        C_in = C_stem
        for C_out, stride in zip(channels, strides):
            self.layers.append(PRIMITIVES['SRB_k3'](C_in, C_out, stride, self.snn_params))
            if stride == 1:
                logging.info(f'SRB_k3\t{img_size} x {img_size} x {C_in}\t-->\t{img_size} x {img_size} x {C_out}')
            elif stride == 2:
                logging.info(f'SRB_k3\t{img_size} x {img_size} x {C_in}\t-->\t{img_size//2} x {img_size//2} x {C_out}')
                img_size = img_size // 2
            C_in = C_out
        logging.info(f'-----------------------------------------------')
                
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(2,2),
            NEURON[neuron](init_tau, v_threshold),
        ) 
        logging.info(f'avgpool /2\t{img_size} x {img_size} x {C_in}\t-->\t{img_size//2} x {img_size//2} x {C_in}')
        img_size = img_size // 2

        # FC layer
        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C_in*img_size*img_size, C_last, bias=False),
            NEURON[neuron](init_tau, v_threshold),
        )
        logging.info(f'FC1     \t{C_in*img_size*img_size}\t-->\t{C_last}')

        self.fc_2 = nn.Sequential(
            nn.Linear(C_last, num_class*10, bias=False), ## before boost
            NEURON[neuron](init_tau, v_threshold),
        )
        logging.info(f'FC2     \t{C_last}\t-->\t{num_class*10}')
        self.boost = nn.AvgPool1d(10, 10)
        logging.info(f'boost   \t{num_class*10}\t-->\t{num_class}')
            
    def forward(self, x):
        num_of_spikes = 0
        out_spikes_counter = None

        x = x.permute(1, 0, 2, 3, 4)

        for t in range(x.shape[0]):
            out = self.spike_neuron_stem(self.conv_stem(x[t]))
            num_of_spikes += out.sum().item()

            out, spikes = self.DVS_max_pool_1(out)
            num_of_spikes += spikes
            out = self.DVS_conv_1(out)
            num_of_spikes += out.sum().item()
            out, spikes = self.DVS_max_pool_2(out)
            num_of_spikes += spikes

            for layer in self.layers:
                out, spikes = layer(out)
                num_of_spikes += spikes
            if self.avgpool is not None:
                out = self.avgpool(out)
                num_of_spikes += out.sum().item()
            out = self.fc_1(out)
            num_of_spikes += out.sum().item()
            out = self.fc_2(out)
            num_of_spikes += out.sum().item()
            if out_spikes_counter is None:
                out_spikes_counter = self.boost(out.unsqueeze(1)).squeeze(1)
            else:
                out_spikes_counter += self.boost(out.unsqueeze(1)).squeeze(1)

        return out_spikes_counter, num_of_spikes / x.shape[1]

""" ref: Going Deeper With Directly-Trained Larger Spiking Neural Networks
"""
def Spiking_DVS_ResNet17_Zheng(num_class, snn_params, init_channels=64):
    c = init_channels
    model_spec = {
        'C_stem': c,
        'channels': [c, c, c, c*2, c*2, c*2, c*2],
        'C_last': c*4, ## FC1,
        'strides': [2, 1, 1, 2, 1, 1, 1],
    }
    return DVS_ResNet(num_class, snn_params, model_spec)

