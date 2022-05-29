import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from spikingjelly.clock_driven import functional
import numpy as np
import readline
import os
import sys
import argparse
import time
import logging

## this repo
sys.path.insert(0, '.')
import data
import utils

from space import CANDIDATE_BLOCKS
from train_supernet.supernet import SpikingNetwork
from search_arch.evolution_algo import EvolutionarySearch

def run():
    parser = argparse.ArgumentParser()
    # init_tau, batch_size, learning_rate, T_max, log_dir, use_plif
    parser.add_argument('--save', type=str, default='EXP')

    # dummy
    parser.add_argument('--arch_sampling', type=str, default='uniform', help='uniform / priority')
    
    # search setting
    parser.add_argument('--supernet', type=str, default=None)
    parser.add_argument('--search_algo', type=str, default='random', help='[just_sampling / random / evolution]')
    parser.add_argument('--num_return_archs', type=int, default=1)

    parser.add_argument('--dataset_dir', type=str, default=None) ## you must set the directory
    parser.add_argument('--dataset_name', type=str, default='CIFAR10')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021) ## seed must to be identical with the seed used in the super-network training
    parser.add_argument('--search_seed', type=int, default=99)
    parser.add_argument('--train_portion', type=float, default=0.8) ## D_train = 80% and D_val = 20% --> using D_val

    parser.add_argument('--search_space', type=str, default='AutoSNN', help='')
    parser.add_argument('--batch_size', type=int, default=96)

    # SNN
    parser.add_argument('--T', type=int) # rate coding time step
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--v_threshold', type=float, default=1.0)
    parser.add_argument('--neuron', type=str, default='PLIF', help='IF / LIF / PLIF / ANN')

    # SNN + Event-based dataset (neuromorphic dataset)
    parser.add_argument('--split_by', type=str)
    parser.add_argument('--normalization', type=str)
    
    # data augmentation cutout
    parser.add_argument('--cutout', action='store_true', default=False)
    parser.add_argument('--cutout_length', type=int, default=16)

    # evoluation search setting
    parser.add_argument('--max_search_iter', type=int, default=20)
    parser.add_argument('--num_pool', type=int, default=20)
    parser.add_argument('--num_mutation', type=int, default=10)
    parser.add_argument('--mutation_prob', type=float, default=0.2)
    parser.add_argument('--num_crossover', type=int, default=10)
    parser.add_argument('--num_topk', type=int, default=10)
    parser.add_argument('--fitness', type=str, default='ACC_pow_spikes', help='ACC / ACC_pow_spikes')
    parser.add_argument('--fitness_lambda', type=float, default=-0.08, help='')
    parser.add_argument('--avg_num_spikes', type=int, default=110000)

    args = parser.parse_args()

    if args.supernet is None:
       raise ValueError('A trained supernet is required.')

    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name

    T = args.T
    init_tau = args.init_tau
    v_threshold = args.v_threshold
    neuron = args.neuron

    split_by = args.split_by
    normalization = args.normalization
    if normalization == 'None':
        normalization = None

    args.save = args.supernet.replace('checkpoint.pth.tar', '')
    tmp_neuron = 'ANN' if args.neuron == 'ANN' else 'SNN'
    if args.search_algo == 'just_sampling':
        search_name = f'{args.search_space}_{args.dataset_name}_{tmp_neuron}_{args.search_algo}_{args.seed}'
    else:
        search_name = f'{args.search_space}_{args.dataset_name}_{tmp_neuron}_{args.seed}_{args.search_algo}_{args.fitness}_{args.fitness_lambda}_{args.search_seed}'
        arch_prefix = f'{args.search_space}_{args.dataset_name}_{tmp_neuron}_{args.seed}_{args.search_algo}_{args.fitness}_{-1*int(args.fitness_lambda*100)}_{args.search_seed}'

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'search_{}_log_{}.txt').format(search_name, time.strftime("%Y%m%d-%H%M%S")))
    #fh = logging.FileHandler(os.path.join(args.save, 'debug.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('Experiment dir : {}'.format(args.save))
    for arg, val in args.__dict__.items():
      logging.info(arg + '.' * (60 - len(arg) - len(str(val))) + str(val))   

    # set randomness
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)

    # get data loaders
    train_loader, valid_loader, args.num_class = data.get_train_val_loaders(args, search=True)

    # build model
    snn_params={'T':            T, 
                'init_tau':     init_tau, 
                'v_threshold':  v_threshold,
                'neuron':       neuron,
                'is_DVS_data':  'DVS' in args.dataset_name,
    }
    net = SpikingNetwork(args.search_space, args.num_class, snn_params, args)

    #device = torch.device(f"cuda:{args.gpu}")
    ckpt = torch.load(args.supernet, map_location=torch.device('cpu'))
    net.load_state_dict(ckpt['net'])

    net = net.cuda()

    ## after data split (using super-network training seed), a new seed is used
    np.random.seed(args.search_seed)
    torch.manual_seed(args.search_seed)
    torch.cuda.manual_seed(args.search_seed)

    if args.search_algo == 'just_sampling':
        # just sampling for experiment to check rank correlation
        just_sampling(net, valid_loader, args)
        return
    else:
        worker = EvolutionarySearch(args, net)
        history = worker.search(args.max_search_iter, valid_loader, train_loader) 

    # save the history
    torch.save(history, os.path.join(args.save, f'search_{search_name}_history.pth'))

    # from last top-k, return return-k architectures
    topk_pool = history[-1][:args.num_return_archs]
    arch_file = open('search_arch/arch.py', 'a')
    arch_file.write(f'# {args.save}\n')
    arch_file.write(f'# lmabda = {args.fitness_lambda}\n')
    k = 1
    for arch, acc, spikes, fitness in topk_pool:
        arch_file.write(f'# val acc: {acc:.4f} spikes: {spikes:.0f} fitness: {fitness:0.4f}\n')
        arch_name = f'{arch_prefix}_{k}'
        arch_blocks = ' = ['
        for i, idx in enumerate(arch):
            if idx == -1:
                arch_blocks += f'\'max_pool_k2\''
            else:
                arch_blocks += f'\'{CANDIDATE_BLOCKS[idx]}\''
            if i < len(arch)-1:
                arch_blocks += ', '
        arch_blocks += ']\n'
        arch_file.write(arch_name + arch_blocks)
        k += 1

    arch_file.close()


def just_sampling(net, valid_loader, args):
    arch_file = open('search_arch/arch.py', 'a')
    nn = 1
    for i in range(nn, nn+10):
        random_block_ids = net._uniform_sampling()
        logging.info(f'{i}: {random_block_ids}')
        arch_file.write(f'{args.search_space}_random_{i} = [')
        for i, idx in enumerate(random_block_ids):
            if idx == -1:
                arch_file.write(f'\'max_pool_k2\'')
            else:
                arch_file.write(f'\'{CANDIDATE_BLOCKS[idx]}\'')
            if i < len(random_block_ids) -1:
                arch_file.write(',')
        arch_file.write(']\n')

    arch_file.close()


if __name__ == '__main__':
    run()
