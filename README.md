# AutoSNN: Towards Energy-Efficient Spiking Neural Networks 

## Introduction
This is the official code of our paper, [AutoSNN: Towards Energy-Efficient Spiking Neural Networks](arxiv.org/abs/2201.12738), accepted in [ICML 2022](icml.cc).

## Our experimental environment
```
Python >= 3.6.10, PyTorch == 1.4.0, torchvision == 0.5.0
```
For training and evaluating SNNs, we used one of old versions of [spikingjelly](https://github.com/fangwei123456/spikingjelly), which can be installed as follows:
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
git reset --hard 73f94ab983d0167623015537f7d4460b064cfca1
python setup.py install
```

## Datasets
CIFAR-10, CIFAR-100, and SVHN can be automatically downloaded by torchvision, but Tiny-ImageNet needs to be manually downloaded.

Neuromorphic datasets (CIFAR10-DVS, DVS128Gesture) can be downloaded by using [this link](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.4/spikingjelly.datasets.html)

## Search
Our method has two-step search processes: training a super-network and searching for SNNs, which can be executed with `1_script_train_supernet.sh` and `2_script_search.sh`, respectively.

## Retraining
After the search process, SNN architectures searched by our method will be automatically saved in `search_arch/arch.py`.

We provide a script `3_script_retrain.sh` to train the searched SNN architectures.
