import torch
import numpy as np
import logging
from copy import deepcopy

from spikingjelly.clock_driven import functional

import utils
from space import CANDIDATE_BLOCKS, MACRO_SEARCH_SPACE

class EvolutionarySearch(object):

    def __init__(self, args, net):
        super(EvolutionarySearch, self).__init__()

        self.num_blocks = len(CANDIDATE_BLOCKS)

        self.args = args
        self.net = net
        self.search_space = MACRO_SEARCH_SPACE[args.search_space]
        self.channels = self.search_space['block_channels']
        self.strides = self.search_space['strides']

        self.num_pool = args.num_pool
        self.num_mutation = args.num_mutation
        self.mutation_prob = args.mutation_prob
        self.num_crossover = args.num_crossover
        
        self.num_topk = args.num_topk
        self.history_topk = [] # including top-k candidates at every iteration during the search

        self.seen_archs = []

        self.fitness_type = args.fitness
        self.fitness_lambda = args.fitness_lambda
        self.spikes_dominator = args.avg_num_spikes

    def _random_pool(self, pool, pool_size):
        new_pool = []
        while len(new_pool) < pool_size:
            cand_arch = self.net._uniform_sampling().tolist()
            if (cand_arch not in new_pool) and (cand_arch not in self.seen_archs):
                new_pool.append(cand_arch)

        return new_pool

    def _mutation(self, pool, pool_size):
        new_pool = []
        for _ in range(pool_size * 3): # time limit
            cand_arch = deepcopy(pool[np.random.choice(range(len(pool)), 1)[0]])
            num_mut = 0
            for layer in range(len(cand_arch)):
                if self.channels[layer] == 'm' and self.strides[layer] == 2:
                    # for max_pool_k2
                    cand_arch[layer] = -1
                else:
                    if np.random.random() < self.mutation_prob:
                        choices = [i for i in range(self.num_blocks) if i != cand_arch[layer]]
                        cand_arch[layer] = np.random.choice(choices, 1)[0]
                        num_mut += 1

            if (cand_arch not in new_pool) and (cand_arch not in self.seen_archs):
                new_pool.append(cand_arch)

            if len(new_pool) == pool_size:
                break

        return new_pool

    def _crossover(self, pool, pool_size):
        new_pool = []
        for _ in range(pool_size * 3): # time limit
            parent_1, parent_2 = np.random.choice(range(len(pool)), 2, replace=False)
            parent_1 = pool[parent_1]
            parent_2 = pool[parent_2]
            
            division_pos = np.random.choice(range(1, len(parent_1)), 1)[0]
            cand_arch = parent_1[:division_pos] + parent_2[division_pos:]

            if (cand_arch not in new_pool) and (cand_arch not in self.seen_archs):
                new_pool.append(cand_arch)

            if len(new_pool) == pool_size:
                break

        return new_pool


    def search(self, max_search_iter, loader, train_loader):
        # first pool
        pool = self._random_pool([], self.num_pool) 
        for arch in pool:
            self.seen_archs.append(arch)

        topk_pool = []
        topk_acc = []
        topk_spikes = []
        topk_fitness = []
        for it in range(max_search_iter):
            logging.info(f'search_iter: {it}')
            acc_list = []
            spikes_list = []
            fitness_list = []
            for idx, arch in enumerate(pool):
                acc, spikes = self.infer(loader, self.net, self.args, arch)
                acc_list.append(acc)
                spikes_list.append(spikes)
                if self.fitness_type == 'ACC':
                    fitness = acc 
                elif self.fitness_type == 'ACC_pow_spikes':
                    fitness = acc * pow(spikes / self.spikes_dominator, self.fitness_lambda)
                fitness_list.append(fitness)
                logging.info(f'{idx} {arch} {acc:.4f} {spikes:.0f} {fitness:.4f}')
        

            # get top-k candidates
            tmp_pool = topk_pool + pool
            tmp_acc = topk_acc + acc_list
            tmp_spikes = topk_spikes + spikes_list
            tmp_fitness = topk_fitness + fitness_list

            topk_idx = np.argsort(tmp_fitness)[::-1][:self.num_topk] ## decreasing order: the first is the highest
            arch_acc_spikes_fitness = [ [tmp_pool[idx], tmp_acc[idx], tmp_spikes[idx], tmp_fitness[idx]] for idx in topk_idx ]
            topk_pool = [ tmp_pool[idx] for idx in topk_idx ]
            topk_acc = [ tmp_acc[idx] for idx in topk_idx ]
            topk_spikes = [ tmp_spikes[idx] for idx in topk_idx ]
            topk_fitness = [ tmp_fitness[idx] for idx in topk_idx ]

            logging.info(f'\ttop-{self.num_topk} paths')
            for arch, acc, spikes, fitness in zip(topk_pool, topk_acc, topk_spikes, topk_fitness):
                logging.info(f'\t{arch} {acc:.4f} {spikes:.0f} {fitness:.4f}')

            # save history
            self.history_topk.append(arch_acc_spikes_fitness)
            if it == max_search_iter -1:
                break

            # prepare next pool
            if self.args.search_algo == 'random':
                logging.info(f'[new pool] random: {self.num_pool}')
                pool = self._random_pool(topk_pool, self.num_pool)
                for arch in pool:
                    self.seen_archs.append(arch)
            elif self.args.search_algo == 'evolution':
                mut_pool = self._mutation(topk_pool, self.num_mutation)
                for arch in mut_pool:
                    self.seen_archs.append(arch)
                cro_pool = self._crossover(topk_pool, self.num_crossover)
                for arch in cro_pool:
                    self.seen_archs.append(arch)
                rnd_pool = self._random_pool(topk_pool, self.num_pool - len(mut_pool) - len(cro_pool))
                for arch in rnd_pool:
                    self.seen_archs.append(arch)
                logging.info(f'[new pool] mutation: {len(mut_pool)}, crossover: {len(cro_pool)}, random: {len(rnd_pool)}')
                pool = mut_pool + cro_pool + rnd_pool

        # return the history
        return self.history_topk

    def infer(self, loader, net, args, block_ids=None):
        assert(block_ids is not None)
        top1 = utils.AverageMeter()
        num_spikes = utils.AverageMeter()
        net.eval()

        total_correct = 0
        total_num = 0
        num_steps = len(loader)
        with torch.no_grad():
            for step, (input, target) in enumerate(loader):
                input = input.cuda()
                target = target.cuda()

                if args.neuron == 'ANN':
                    logits, _ = net(input, block_ids)
                    acc, _ = utils.accuracy(logits, target, topk=(1, 5))
                    num_of_spikes = 0
                else:
                    out_spikes_counter, num_of_spikes, _ = net(input, block_ids)
                    functional.reset_net(net)
                    acc = (out_spikes_counter.argmax(dim=1) == target).float().sum()
                    total_correct += acc

                n = input.size(0)
                total_num += n
                top1.update(acc.item(), n)
                num_spikes.update(num_of_spikes, n)

        if args.neuron == 'ANN':
            return top1.avg, num_spikes.avg
        else:
            return total_correct / total_num, num_spikes.avg

