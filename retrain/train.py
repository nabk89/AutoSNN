import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import readline
import os
import sys
import argparse
import time
import logging
from thop import profile

# for SNN
from spikingjelly.clock_driven import functional

## this repo
sys.path.insert(0, '.')
import data
import utils

# conventional network
from conventional_models import Spiking_CIFARNet_Wu
from conventional_models import Spiking_CIFARNet_Fang
from conventional_models import Spiking_ResNet19_Zheng
from conventional_models import Spiking_ResNet11_Lee
from conventional_models import Spiking_DVS_CIFARNet_Fang
from conventional_models import Spiking_DVS_ResNet17_Zheng

# Variation (GAP)
from conventional_models import Spiking_ResNet19_Zheng_no_GAP
from conventional_models import Spiking_ResNet19_Zheng_GAP

# AutoSNN architeture
from retrain.childnet import SpikingNetwork

# network specification
from search_arch import arch

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='EXP')
    parser.add_argument('--report_freq', type=int, default=50)
    
    # Conventional architectures
    parser.add_argument('--conventional_arch', type=str, default=None)
    parser.add_argument('--conventional_init_channels', type=int, default=128)

    # Architectures searched by AutoSNN
    parser.add_argument('--macro_type', type=str, default='AutoSNN_16', help='AutoSNN_16 / AutoSNN_32 / AutoSNN_64 / AutoSNN_128 / SNN_2 / SNN_3 / SNN_4')
    parser.add_argument('--arch', type=str, default=None)

    # training options
    parser.add_argument('--dataset_dir', type=str, default=None) ## you must set the directory
    parser.add_argument('--dataset_name', type=str, default='CIFAR10')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    # SNN (spikingjelly)
    parser.add_argument('--T', type=int) # rate coding time step
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--v_threshold', type=float, default=1.0)
    parser.add_argument('--neuron', type=str, default='PLIF', help='IF / LIF / PLIF')

    # SNN + Event-based dataset (neuromorphic dataset)
    parser.add_argument('--split_by', type=str, default='number')
    parser.add_argument('--normalization', type=str, default='None')

    # data augmentation: cutout
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
        args.normalization = None

    if 'DVS' in dataset_name:
        args.save = f'retrain_result/{args.save}/{args.macro_type}_{dataset_name}_{args.optimizer}_{args.seed}_T_{T}_init_tau_{init_tau}_vth_{v_threshold}_neuron_{neuron}_split_by_{split_by}_normalization_{normalization}_cutout_{args.cutout}'
    else:
        if args.neuron == 'ANN':
            args.save = f'retrain_result/{args.save}/{args.macro_type}_{dataset_name}_{args.optimizer}_{args.seed}_{neuron}_cutout_{args.cutout}'
        else:
            assert((args.conventional_arch is None) ^ (args.arch is None))
            if args.conventional_arch is None:
                args.save = f'retrain_result/{args.save}/{args.macro_type}_{dataset_name}_{args.optimizer}_{args.seed}_T_{T}_init_tau_{init_tau}_vth_{v_threshold}_neuron_{neuron}_cutout_{args.cutout}'
            else:
                args.save = f'retrain_result/{args.save}/{args.conventional_init_channels}_{dataset_name}_{args.optimizer}_{args.seed}_T_{T}_init_tau_{init_tau}_vth_{v_threshold}_neuron_{neuron}_cutout_{args.cutout}'

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
    train_loader, valid_loader, args.num_class = data.get_train_val_loaders(args)

    ## check network size (+ FLOPs which is not meaningless in SNN)
    net = load_network(args) # --> dummy network!
    if 'DVS' in args.dataset_name:
        img_size = 128
        img_ch = 2
        flops, params = profile(net, inputs=(torch.randn(1, args.T, img_ch, img_size, img_size),), verbose=False)
    else:
        if 'Tiny-ImageNet-200' == args.dataset_name:
            img_size = 64
        else:
            img_size = 32
        img_ch = 3
        flops, params = profile(net, inputs=(torch.randn(1, img_ch, img_size, img_size),), verbose=False)
    logging.info("="*60)
    logging.info(f"thop profile result: params = {params/1e6:.3f}M, flops = {flops/1e6:.3f}M\n")
    del net

    ## load the real network
    net = load_network(args)
    net = net.cuda()

    if args.optimizer == 'SGD': 
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9,weight_decay=3e-4)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * args.epochs), int(0.75 * args.epochs)], last_epoch=-1)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    if args.neuron == 'ANN':
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
    else:
        criterion = None

    start_epoch = 0
    is_best = False
    best_acc = 0
    best_acc_epoch =0
    best_acc_spikes = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint {args.resume}")
            device = torch.device(f"cuda:{args.gpu}")
            ckpt = torch.load(args.resume, map_location=device)
            start_epoch = ckpt['epoch']
            net.load_state_dict(ckpt['net'])
            best_acc = ckpt['best_acc']
            best_acc_epoch = ckpt['best_acc_epoch']
            best_acc_spikes = ckpt['best_acc_spikes']
            optimizer.load_state_dict(ckpt['optimizer'])
            if args.optimizer == 'SGD':
                scheduler.load_state_dict(ckpt['scheduler'])
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        if args.optimizer == 'SGD':
            logging.info(f'epoch {epoch} lr {scheduler.get_lr()[0]}')
        else:
            logging.info(f'epoch {epoch}')

        # training
        train_acc, train_obj, train_avg_spikes = train(train_loader, net, optimizer, args, criterion)
        logging.info(f'train_acc {train_acc:.6f} train_avg_spikes {train_avg_spikes:.0f}')

        # validation
        valid_acc, val_avg_spikes = infer(valid_loader, net, args, criterion)
        is_best = valid_acc > best_acc
        if is_best:
            best_acc = valid_acc
            best_acc_epoch = epoch
            best_acc_spikes = val_avg_spikes
            torch.save(net.state_dict(), os.path.join(args.save, 'weight_best.pt'))
        logging.info(f'valid_acc {valid_acc:.6f} val_avg_spikes {val_avg_spikes:.0f} best_acc {best_acc:.6f} best_acc_spikes {best_acc_spikes:.0f} (at epoch {best_acc_epoch})')

        if args.optimizer == 'SGD':
            scheduler.step()

        if args.optimizer == 'SGD':
          torch.save({
            'epoch': epoch+1,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()}, os.path.join(args.save, 'checkpoint.pth.tar'))
        else:
          torch.save({
            'epoch': epoch+1,
            'net': net.state_dict(),
            'best_acc': best_acc,
            'best_acc_epoch': best_acc_epoch,
            'best_acc_spikes': best_acc_spikes,
            'optimizer': optimizer.state_dict()}, os.path.join(args.save, 'checkpoint.pth.tar'))
        logging.info('epoch time: %d sec.', time.time() - epoch_start)


def train(loader, net, optimizer, args, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    num_spikes = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    net.train()

    cutout = None
    if args.cutout and 'DVS' in args.dataset_name:
        cutout = data.Cutout(args.cutout_length, 1.0)

    num_steps = len(loader)
    for step, (input, target) in enumerate(loader):
        batch_start = time.time()

        if cutout is not None:
            input = cutout(input)

        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        if args.neuron == 'ANN':
            logits = net(input)
            loss = criterion(logits, target)
            num_of_spikes = 0
        else:
            out_spikes_counter, num_of_spikes = net(input)
            out_spikes_counter_frequency = out_spikes_counter / net.T
            loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(target, args.num_class).float())
            #loss = criterion(out_spikes_counter_frequency, target)
        loss.backward()
        optimizer.step()
        functional.reset_net(net)

        if args.neuron == 'ANN':
            acc, _ = utils.accuracy(logits, target, topk=(1, 5))
        else:
            acc = (out_spikes_counter_frequency.argmax(dim=1) == target).float().mean()
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(acc.item(), n)
        num_spikes.update(num_of_spikes, n)
        batch_time.update(time.time() - batch_start)

        if step % args.report_freq == 0 or step == num_steps-1:
            logging.info('train (%03d/%d) loss: %e top1: %f batchtime: %.2f sec', step, num_steps, objs.avg, top1.avg, batch_time.avg)
    return top1.avg, objs.avg, num_spikes.avg


def infer(loader, net, args, criterion):
    top1 = utils.AverageMeter()
    num_spikes = utils.AverageMeter()
    net.eval()

    total_correct = 0
    total_num = 0
    num_steps = len(loader)
    with torch.no_grad():
        for step, (input, target) in enumerate(loader):
            batch_start = time.time()
            input = input.cuda()
            target = target.cuda()

            if args.neuron == 'ANN':
                logits = net(input)
                loss = criterion(logits, target)
                acc, _ = utils.accuracy(logits, target, topk=(1, 5))
                num_of_spikes = 0
            else:
                out_spikes_counter, num_of_spikes = net(input)
                functional.reset_net(net)
                #acc = (out_spikes_counter.argmax(dim=1) == target).float().mean()
                acc = (out_spikes_counter.argmax(dim=1) == target).float().sum()
                total_correct += acc
                
            n = input.size(0)
            total_num += n
            top1.update(acc.item(), n)
            num_spikes.update(num_of_spikes, n)

            if step % args.report_freq == 0 or step == num_steps-1:
                logging.info('valid (%03d/%d) top1: %f', step, num_steps, total_correct/total_num)

    if args.neuron == 'ANN':
        return top1.avg, num_spikes.avg
    else:
        return total_correct / total_num, num_spikes.avg


def load_network(args):
    snn_params={'T':            args.T, 
                'init_tau':     args.init_tau,
                'v_threshold':  args.v_threshold,
                'neuron':       args.neuron,
                'is_DVS_data':  'DVS' in args.dataset_name,
    }
    if args.conventional_arch is None:
        ## backbone (macro_type) + searched architecture by AutoSNN
        net = SpikingNetwork(args.macro_type, args.num_class, snn_params, arch=eval(f"arch.{args.arch}"), args=args)
    else:
        ## predefined conventional network
        if args.conventional_arch == 'CIFARNet_Wu':
            net = Spiking_CIFARNet_Wu(args.num_class, snn_params, args.conventional_init_channels)
        elif args.conventional_arch == 'CIFARNet_Fang':
            net = Spiking_CIFARNet_Fang(args.num_class, snn_params, args.conventional_init_channels)
        elif args.conventional_arch == 'ResNet19_Zheng':
            net = Spiking_ResNet19_Zheng(args.num_class, snn_params, args.conventional_init_channels)
        elif args.conventional_arch == 'ResNet19_Zheng_no_GAP':
            net = Spiking_ResNet19_Zheng_no_GAP(args.num_class, snn_params, args.conventional_init_channels)
        elif args.conventional_arch == 'ResNet19_Zheng_GAP':
            net = Spiking_ResNet19_Zheng_GAP(args.num_class, snn_params, args.conventional_init_channels)
        elif args.conventional_arch == 'ResNet11_Lee':
            net = Spiking_ResNet11_Lee(args.num_class, snn_params, args.conventional_init_channels)
        elif args.conventional_arch == 'DVS_CIFARNet_Fang':
            if args.dataset_name == 'CIFAR10DVS':
                num_layers = 4
            elif args.dataset_name == 'DVS128Gesture':
                num_layers = 5
            net = Spiking_DVS_CIFARNet_Fang(args.num_class, num_layers, snn_params, args.conventional_init_channels)
        elif args.conventional_arch == 'DVS_ResNet17_Zheng':
            net = Spiking_DVS_ResNet17_Zheng(args.num_class, snn_params, args.conventional_init_channels)

    return net

if __name__ == '__main__':
    run()
