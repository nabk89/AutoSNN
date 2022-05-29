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
from train_supernet.supernet import SpikingNetwork


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='EXP')
    parser.add_argument('--report_freq', type=int, default=50)

    parser.add_argument('--dataset_dir', type=str, default=None) ## you must set the directory
    parser.add_argument('--dataset_name', type=str, default='CIFAR10')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--resume', type=str, default=None)

    # super-network training
    parser.add_argument('--search_space', type=str, default='AutoSNN_16', help='')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_portion', type=float, default=0.8) ## D_train = 80% and D_val = 20% --> using D_train
    
    # SNN
    parser.add_argument('--T', type=int, default=8) # rate coding time step
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--v_threshold', type=float, default=1.0)
    parser.add_argument('--neuron', type=str, default='PLIF', help='IF / LIF / PLIF')

    # SNN + Event-based dataset (neuromorphic dataset)
    parser.add_argument('--split_by', type=str, default='number')
    parser.add_argument('--normalization', type=str, default='None')

    # data augmentation
    parser.add_argument('--cutout', action='store_true', default=False)
    parser.add_argument('--cutout_length', type=int, default=16)

    args = parser.parse_args()

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

    if 'DVS' in dataset_name:
        args.save = f'retrain_result/{args.save}/{args.macro_type}_{dataset_name}_{args.optimizer}_{args.seed}_T_{T}_init_tau_{init_tau}_vth_{v_threshold}_neuron_{neuron}_split_by_{split_by}_normalization_{normalization}'
    else:
        if args.neuron == 'ANN':
            args.save = f'macro_search_result/{args.save}/{args.search_space}_{dataset_name}_ANN_{args.optimizer}_{args.epochs}ep_{args.seed}'
        else:
            args.save = f'macro_search_result/{args.save}/{args.search_space}_{dataset_name}_SNN_{args.optimizer}_{args.epochs}ep_{args.seed}'

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log_{}.txt').format(time.strftime("%Y%m%d-%H%M%S")))
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
    train_loader, _, args.num_class = data.get_train_val_loaders(args, search=True)

    # build model
    snn_params={'T':            T, 
                'init_tau':     init_tau, 
                'v_threshold':  v_threshold,
                'neuron':       neuron,
                'is_DVS_data':  'DVS' in args.dataset_name,
    }
    net = SpikingNetwork(args.search_space, args.num_class, snn_params, args)
    net = net.cuda()

    # set optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    
    # set loss function
    if args.neuron == 'ANN':
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
    else:
        criterion = None

    # start training
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint {args.resume}")
            device = torch.device(f"cuda:{args.gpu}")
            ckpt = torch.load(args.resume, map_location=device)
            start_epoch = ckpt['epoch']
            net.load_state_dict(ckpt['net'])
            optimizer.load_state_dict(ckpt['optimizer'])
 
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        head_log = f'epoch {epoch}'
        logging.info(head_log)

        # training
        train_acc, avg_num_spikes, train_obj = train(train_loader, net, optimizer, args, criterion)
        logging.info(f'train_acc {train_acc:.6f} avg_num_spikes {avg_num_spikes}')

        save_dict = {
            'epoch': epoch+1,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'avg_num_spikes': avg_num_spikes,
        } 

        torch.save(save_dict, os.path.join(args.save, 'checkpoint.pth.tar'))
        logging.info('epoch time: %d sec.', time.time() - epoch_start)

def train(loader, net, optimizer, args, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    num_spikes = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    net.train()

    total_correct = 0
    total_num = 0
    num_steps = len(loader)
    for step, (input, target) in enumerate(loader):
        batch_start = time.time()
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        if args.neuron == 'ANN':
            logits, block_ids = net(input)
            loss = criterion(logits, target)
            num_of_spikes = 0
        else:
            out_spikes_counter, num_of_spikes, block_ids = net(input)
            out_spikes_counter_frequency = out_spikes_counter / net.T
            loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(target, args.num_class).float())
        loss.backward()
        optimizer.step()

        if args.neuron == 'ANN':
            acc, _ = utils.accuracy(logits, target, topk=(1, 5))
        else:
            functional.reset_net(net)
            acc = (out_spikes_counter.argmax(dim=1) == target).float().mean()
            total_correct += (out_spikes_counter.argmax(dim=1) == target).float().sum()

        n = input.size(0)
        total_num += n
        objs.update(loss.item(), n)
        top1.update(acc.item(), n)
        num_spikes.update(num_of_spikes, n)

        batch_time.update(time.time() - batch_start)
        if step % args.report_freq == 0 or step == num_steps-1:
            logging.info('train (%03d/%d) loss: %e top1: %f batchtime: %.2f sec', step, num_steps, objs.avg, top1.avg, batch_time.avg)
    if args.neuron == 'ANN':
        return top1.avg, 0, objs.avg
    else:
        return total_correct/total_num, num_spikes.avg, objs.avg


if __name__ == '__main__':
    run()
