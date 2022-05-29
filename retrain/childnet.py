import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from blocks import PRIMITIVES
from space import MACRO_SEARCH_SPACE
from plif_node import NEURON

import logging

class SpikingNetwork(nn.Module):

    def __init__(self, search_space, num_class, snn_params, arch=None, args=None):
        super(SpikingNetwork, self).__init__()

        assert(arch is not None)

        # SNN parameters
        self.snn_params = snn_params
        if snn_params is not None:
            self.T = snn_params['T']
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']
            self.is_DVS_data = snn_params['is_DVS_data']
            if neuron == 'ANN':
                self.T = 1

        # build network
        # load macro search space
        self.search_space = MACRO_SEARCH_SPACE[search_space]

        C_stem   = self.search_space['stem_channel']
        channels = self.search_space['block_channels']
        strides  = self.search_space['strides']
        use_GAP  = self.search_space['use_GAP']

        self.is_tiny_imagenet = False
        if self.is_DVS_data:
            img_size = 128
            C_in =  2
        else:
            C_in = 3
            if args.dataset_name == 'Tiny-ImageNet-200':
                img_size = 64
                self.is_tiny_imagenet = True
            else:
                img_size = 32

        # stem conv layer
        self.conv_stem = nn.Sequential(
            nn.Conv2d(C_in, C_stem, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )
        self.spike_neuron_stem = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        logging.info(f'stem conv 3x3\t{img_size} x {img_size} x {C_in}\t-->\t{img_size} x {img_size} x {C_stem}')

        # additional stem layers
        if self.is_DVS_data:
            # for DVS data (input resolution: 128x128)
            logging.info(f'------- additional stem layers for DVS data -------')
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
        elif self.is_tiny_imagenet:
            # for Tiny-ImageNet-200 (input resolution: 64x64)
            logging.info(f'------- additional stem layers for Tiny-ImageNet-200 data -------')
            self.ImageNet_max_pool = PRIMITIVES['max_pool_k2'](C_stem, C_stem, stride=2, snn_params=self.snn_params)
            logging.info(f'max_pool_k2\t{img_size} x {img_size} x {C_stem}\t-->\t{img_size // 2} x {img_size // 2} x {C_stem}')

            img_size = img_size // 2
            self.ImageNet_conv = nn.Sequential(
                nn.Conv2d(C_stem, C_stem, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(C_stem),
                NEURON[neuron](init_tau, v_threshold),
            )
            logging.info(f'conv_k3   \t{img_size} x {img_size} x {C_stem}\t-->\t{img_size} x {img_size} x {C_stem}')


        # block layers (searched from supernet)
        logging.info(f'------------------- block layers -------------------')
        self.layers = nn.ModuleList()
        C_in = C_stem
        for C_out, stride, block_name in zip(channels, strides, arch):
            if stride == 1:
                logging.info(f'{block_name}\t{img_size} x {img_size} x {C_in}\t-->\t{img_size} x {img_size} x {C_out}')
            elif stride == 2:
                if C_out == 'm':
                    block_name = 'max_pool_k2'
                    C_out = C_in ## change 'm' -> C_in
                elif C_out == 'a':
                    block_name = 'avg_pool_k2'
                    C_out = C_in ## change 'a' -> C_in
                logging.info(f'{block_name}\t{img_size} x {img_size} x {C_in}\t-->\t{img_size//2} x {img_size//2} x {C_out}')
                img_size = img_size // 2
            self.layers.append(PRIMITIVES[block_name](C_in, C_out, stride, self.snn_params))
            C_in = C_out
        logging.info(f'----------------------------------------------------')

        # FC layer
        if neuron == 'ANN':
            if use_GAP:
                self.avgpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                )
                logging.info(f'GAP   \t{img_size} x {img_size} x {C_in}\t-->\t1 x 1 x {C_in}')
                img_size = 1
            else:
                self.avgpool = None
            C_in = C_in * img_size * img_size

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(C_in, num_class, bias=False)
            )
            logging.info(f'FC    \t{C_in}\t-->\t{num_class*10}')
        else:
            if use_GAP:
                self.avgpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    NEURON[neuron](init_tau, v_threshold),
                )
                logging.info(f'Spike GAP\t{img_size} x {img_size} x {C_in}\t-->\t1 x 1 x {C_in}')
                img_size = 1
            else:
                self.avgpool = None
            C_in = C_in * img_size * img_size

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(C_in, num_class*10, bias=False), ## before boost
                NEURON[neuron](init_tau, v_threshold),
            )
            logging.info(f'Spike FC\t{C_in}\t-->\t{num_class*10}')
                
            self.boost = nn.AvgPool1d(10, 10)
            logging.info(f'boost   \t{num_class*10}\t-->\t{num_class}')
            
    def forward(self, x):
        if self.snn_params['neuron'] == 'ANN':
            out = F.relu(self.conv_stem(x))
            for idx, layer in enumerate(self.layers):
                out, _ = layer(out)
            if self.avgpool is not None:
                out = self.avgpool(out)
            out = self.fc(out)
            return out
        elif self.is_DVS_data:
            out_spikes_counter = None
            num_of_spikes = 0
            x = x.permute(1, 0, 2, 3, 4) # [T, N, 2, *, *]

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
                out = self.fc(out)
                num_of_spikes += out.sum().item()
                if out_spikes_counter is None:
                    out_spikes_counter = self.boost(out.unsqueeze(1)).squeeze(1)
                else:
                    out_spikes_counter += self.boost(out.unsqueeze(1)).squeeze(1)

            return out_spikes_counter, num_of_spikes / x.shape[1] ## batch size N
        else:
            num_of_spikes = 0
            out_spikes_counter = None
            
            x = self.conv_stem(x)
            for t in range(self.T):
                out = self.spike_neuron_stem(x)
                spikes = out.sum().item()
                num_of_spikes += spikes

                if self.is_tiny_imagenet:
                    out, spikes = self.ImageNet_max_pool(out)
                    num_of_spikes += spikes
                    out = self.ImageNet_conv(out)
                    num_of_spikes += out.sum().item()
 
                for idx, layer in enumerate(self.layers):
                    out, spikes = layer(out)
                    num_of_spikes += spikes

                if self.avgpool is not None:
                    out = self.avgpool(out)
                    spikes = out.sum().item()
                    num_of_spikes += spikes

                out = self.fc(out)
                spikes = out.sum().item()
                num_of_spikes += spikes

                if out_spikes_counter is None:
                    out_spikes_counter = self.boost(out.unsqueeze(1)).squeeze(1)
                else:
                    out_spikes_counter += self.boost(out.unsqueeze(1)).squeeze(1)

            return out_spikes_counter, num_of_spikes / x.shape[0]

