import os
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch
import numpy as np
import logging

class Cutout(object):
    def __init__(self, length, prob=1.0):
      self.length = length
      self.prob = prob

    def __call__(self, img):
      if np.random.binomial(1, self.prob):
        h, w = img.size(-2), img.size(-1)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
      return img

def get_transforms(dataset_name):
    transform_train = None
    transform_test = None
    if dataset_name == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])
    elif dataset_name == 'SVHN':
        SVHN_MEAN = [0.4377, 0.4438, 0.4728]
        SVHN_STD = [0.1980, 0.2010, 0.1970] 
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
    elif dataset_name == 'CIFAR10':
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] 
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    elif dataset_name == 'CIFAR100':
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    elif dataset_name == 'Tiny-ImageNet-200':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])
    elif dataset_name == 'CIFAR10DVS' or dataset_name == 'DVS128Gesture':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    return transform_train, transform_test

def get_train_val_loaders(args, search=False):
    dataset_name = args.dataset_name
    dataset_dir = args.dataset_dir
    transform_train, transform_test = get_transforms(dataset_name)
    if args.cutout:
        transform_train.transforms.append(Cutout(args.cutout_length, 1.0))

    class_num = 10
    if dataset_name == 'MNIST':
        train_data = torchvision.datasets.MNIST(root=dataset_dir, train=True, transform=transform_train, download=True)
        test_data = torchvision.datasets.MNIST(root=dataset_dir, train=False, transform=transform_test, download=True)
    elif dataset_name == 'SVHN':
        train_data = torchvision.datasets.SVHN(root=dataset_dir, split='train', transform=transform_train, download=True)
        test_data = torchvision.datasets.SVHN(root=dataset_dir, split='test', transform=transform_train, download=True)
    elif dataset_name == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform_train, download=True)
        test_data = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, transform=transform_test, download=True)
    elif dataset_name == 'CIFAR100':
        class_num = 100
        train_data = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, transform=transform_train, download=True)
        test_data = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, transform=transform_test, download=True)
    elif dataset_name == 'Tiny-ImageNet-200':
        class_num = 200
        train_data = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'tiny-imagenet-200/train'), transform_train)
        test_data = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'tiny-imagenet-200/val'), transform_test)
    elif dataset_name == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        train_data = CIFAR10DVS(dataset_dir, train=True, split_ratio=0.9, use_frame=True, frames_num=args.T, split_by=args.split_by, normalization=args.normalization)
        test_data = CIFAR10DVS(dataset_dir, train=False, split_ratio=0.9, use_frame=True, frames_num=args.T, split_by=args.split_by, normalization=args.normalization)
    elif dataset_name == 'DVS128Gesture':
        class_num = 11
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        train_data = DVS128Gesture(dataset_dir, train=True, use_frame=True, frames_num=args.T, split_by=args.split_by, normalization=args.normalization)
        test_data = DVS128Gesture(dataset_dir, train=False, use_frame=True, frames_num=args.T, split_by=args.split_by, normalization=args.normalization)

    # make loaders
    if search:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        logging.info('D_train: %d, D_val: %d'%(split, num_train - split))
        train_idx, valid_idx = indices[:split], indices[split:]

        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=args.batch_size, 
            sampler=SubsetRandomSampler(train_idx), num_workers=4, drop_last=False, pin_memory=True)
        valid_data_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=args.batch_size*8, 
            sampler=SubsetRandomSampler(valid_idx), num_workers=4, drop_last=False, pin_memory=True)

        num_class_train = [0] * class_num
        for data, target in train_data_loader:
            for t in target:
                num_class_train[t] += 1
        num_class_valid = [0] * class_num
        for data, target in valid_data_loader:
            for t in target:
                num_class_valid[t] += 1
        logging.info(f'D_train: {num_class_train}')
        logging.info(f'D_val: {num_class_valid}')
        logging.info('When training the super-network, D_val is not used; When searching, D_train is not used.')

        return train_data_loader, valid_data_loader, class_num
    else:
        # for retraining
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=4, drop_last=False, pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_data, batch_size=args.batch_size*8, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True)
        return train_data_loader, test_data_loader, class_num
