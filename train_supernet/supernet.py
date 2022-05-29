import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from copy import deepcopy

from blocks import PRIMITIVES
from space import CANDIDATE_BLOCKS, MACRO_SEARCH_SPACE 
from plif_node import NEURON
import utils

import logging

class SpikingBlockSet(nn.Module):
    def __init__(self, C_in, C_out, stride, snn_params):
        super(SpikingBlockSet, self).__init__()
 
        self.blocks = nn.ModuleList()
        for cand in CANDIDATE_BLOCKS:
            self.blocks.append(PRIMITIVES[cand](C_in, C_out, stride, snn_params))
        
    def forward(self, x, block_id):
        return self.blocks[block_id](x)


class SpikingNetwork(nn.Module):

    def __init__(self, search_space, num_class, snn_params, args):
        super(SpikingNetwork, self).__init__()

        # SNN parameters
        self.snn_params = snn_params
        if snn_params is not None:
            self.T = snn_params['T']
            init_tau = snn_params['init_tau']
            v_threshold = snn_params['v_threshold']
            neuron = snn_params['neuron']
            is_DVS_data = snn_params['is_DVS_data']
            if neuron == 'ANN':
                self.T = 1

        # build network
        self.search_space = MACRO_SEARCH_SPACE[search_space]

        C_stem   = self.search_space['stem_channel']
        channels = self.search_space['block_channels']
        strides  = self.search_space['strides']
        use_GAP  = self.search_space['use_GAP']

        img_size=32
        C_in = 3
        if is_DVS_data:
            img_size = 128
            C_in =  2

        # stem conv layer
        self.conv_stem = nn.Sequential(
            nn.Conv2d(C_in, C_stem, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )
        self.spike_neuron_stem = None if neuron == 'ANN' else NEURON[neuron](init_tau, v_threshold)
        logging.info(f'stem conv 3x3\t{img_size} x {img_size} x 3\t-->\t{img_size} x {img_size} x {C_stem}')

        # additional layers for DVS data
        if is_DVS_data:
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

        # TBD blocks
        self.num_TBDs = 0
        logging.info(f'----------------- TBD block layers -----------------')
        self.layers = nn.ModuleList()
        C_in = C_stem
        for C_out, stride in zip(channels, strides):
            if stride == 1:
                self.layers.append(SpikingBlockSet(C_in, C_out, stride, self.snn_params))
                logging.info(f'TBD (normal)\t{img_size} x {img_size} x {C_in}\t-->\t{img_size} x {img_size} x {C_out}')
                self.num_TBDs += 1
            elif stride == 2:
                if C_out == 'm':
                    block_name = 'max_pool_k2'
                    self.layers.append(PRIMITIVES[block_name](C_in, C_out, stride, self.snn_params))
                    C_out = C_in
                    logging.info(f'{block_name}\t{img_size} x {img_size} x {C_in}\t-->\t{img_size//2} x {img_size//2} x {C_out}')
                else:
                    self.layers.append(SpikingBlockSet(C_in, C_out, stride, self.snn_params))
                    logging.info(f'TBD (down)\t{img_size} x {img_size} x {C_in}\t-->\t{img_size//2} x {img_size//2} x {C_out}')
                    self.num_TBDs += 1
                img_size = img_size // 2
            C_in = C_out
        logging.info(f'----------------------------------------------------')

        # FC layer
        if neuron == 'ANN':
            if use_GAP:
                self.avgpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                )
                logging.info(f'GAP     \t{img_size} x {img_size} x {C_in}\t-->\t1 x 1 x {C_in}')
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

        # calculate the search space size (# of candidate architectures in the space)
        self.search_space_size = pow(len(CANDIDATE_BLOCKS), self.num_TBDs)

    def forward(self, x, block_ids=None):
        if block_ids is None:
            block_ids = self._uniform_sampling()

        if self.snn_params['neuron'] == 'ANN':
            out = F.relu(self.conv_stem(x))
            for layer, block_id in zip(self.layers, block_ids):
                if isinstance(layer, SpikingBlockSet):
                    out, _ = layer(out, block_id)
                else: ## max pooling
                    out, _ = layer(out)
            if self.avgpool is not None:
                out = self.avgpool(out)
            out = self.fc(out)
            return out, block_ids
        else:
            num_of_spikes = 0
            out_spikes_counter = None
            
            x = self.conv_stem(x)
            for t in range(self.T):
                out = self.spike_neuron_stem(x)
                num_of_spikes += out.sum().item()
    
                for layer, block_id in zip(self.layers, block_ids):
                    if isinstance(layer, SpikingBlockSet):
                        out, spikes = layer(out, block_id)
                    else: ## max pooling
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

            return out_spikes_counter, num_of_spikes / x.shape[0], block_ids


    def _uniform_sampling(self):
        self.arch_id = random.choices(range(self.search_space_size), k=1)[0]
        arch_id = self.arch_id
        arch_TBD_block_ids = []
        for i in range(self.num_TBDs):
            arch_TBD_block_ids.append(arch_id % len(CANDIDATE_BLOCKS))
            arch_id = arch_id // len(CANDIDATE_BLOCKS)

        strides  = self.search_space['strides']
        channels = self.search_space['block_channels']
        block_ids = []
        tmp = 0
        for c, s in zip(channels, strides):
            if s == 1:
                #block_ids.append(random.choices(len(CANDIDATE_BLOCKS), k=1)[0])
                block_ids.append(arch_TBD_block_ids[tmp])
                tmp += 1
            elif s == 2:
                if c == 'm':
                    block_ids.append(-1) ## -1 means max pooling
                else:
                    #block_ids.append(random.choices(len(CANDIDATE_BLOCKS), k=1)[0])
                    block_ids.append(arch_TBD_block_ids[tmp])
                    tmp += 1
        return np.array(block_ids)

