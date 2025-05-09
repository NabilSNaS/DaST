from __future__ import print_function

# Standard libraries
import argparse
import os
import gc
import sys
import xlwt
import random
import numpy as np

# Import attacks from advertorch
from advertorch.attacks import LinfBasicIterativeAttack, CarliniWagnerL2Attack, L2BasicIterativeAttack
from advertorch.attacks import GradientSignAttack, PGDAttack, L2PGDAttack

# PyTorch core modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR

# Dataset and transform utilities
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp

# Custom model definitions (Net_s, Net_m, Net_l)
from net import Net_s, Net_m, Net_l

# Set random seeds for reproducibility
SEED = 10000
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(10000)

# Argument parser for runtime configurations
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--adv', type=str, help='attack method')
parser.add_argument('--mode', type=str, help='use which model to generate examples: "black", "white", or "dast"')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--target', action='store_true', help='enable targeted attack')
opt = parser.parse_args()

# Set a random seed if not provided
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Turn off cuDNN auto-tuning for deterministic results
cudnn.benchmark = False

# Warn if CUDA is available but not being used
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Load MNIST test dataset
testset = torchvision.datasets.MNIST(root='~/rm46_scratch/dataset/', train=False,
                                     download=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))

# Create test DataLoader
testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=16)

# Set the device to GPU or CPU
device = torch.device("cuda:0" if opt.cuda else "cpu")

# ----------------------------
# Adversarial Evaluation Logic
# ----------------------------
def test_adver(net, tar_net, attack, target):
    net.eval()
    tar_net.eval()

    # Define the attack method based on argument
    if attack == 'BIM':
        adversary = L2BasicIterativeAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=4.8,
            nb_iter=500, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
    elif attack == 'PGD':
        adversary = L2PGDAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=5.0,
            nb_iter=500, eps_iter=0.2, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
    elif attack == 'FGSM':
        adversary = GradientSignAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.2,
            targeted=opt.target)
    elif attack == 'CW':
        adversary = CarliniWagnerL2Attack(
            net,
            num_classes=10,
            learning_rate=0.05,
            binary_search_steps=10,
            max_iterations=100,
            targeted=opt.target)

    # Evaluate accuracy on clean test data
    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        net.eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
        print('Accuracy of the network on netD: %.2f %%' %
              (100. * correct_netD.float() / total))

    # Evaluate attack success rate and L2 distortion
    correct = 0.0
    total = 0.0
    total_L2_distance = 0.0
    att_num = 0.
    acc_num = 0.
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = tar_net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        if target:
            # Targeted attack: use random target labels
            labels = torch.randint(0, 9, (inputs.size(0),)).to(device)
            ones = torch.ones_like(predicted)
            zeros = torch.zeros_like(predicted)
            acc_sign = torch.where(predicted == labels, zeros, ones)
            acc_num += acc_sign.sum().float()

            adv_inputs_ori = adversary.perturb(inputs, labels)
            L2_distance = (adv_inputs_ori - inputs).squeeze()
            L2_distance = (torch.norm(L2_distance, dim=list(range(1, inputs.squeeze().dim())))).data
            L2_distance = L2_distance * acc_sign
            total_L2_distance += L2_distance.sum()

            with torch.no_grad():
                outputs = tar_net(adv_inputs_ori)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                att_sign = torch.where(predicted == labels, ones, zeros)
                att_sign = att_sign + acc_sign
                att_sign = torch.where(att_sign == 2, ones, zeros)
                att_num += att_sign.sum().float()
        else:
            # Untargeted attack: only attack correctly classified examples
            ones = torch.ones_like(predicted)
            zeros = torch.zeros_like(predicted)
            acc_sign = torch.where(predicted == labels, ones, zeros)
            acc_num += acc_sign.sum().float()

            adv_inputs_ori = adversary.perturb(inputs, labels)
            L2_distance = (adv_inputs_ori - inputs).squeeze()
            L2_distance = (torch.norm(L2_distance, dim=list(range(1, inputs.squeeze().dim())))).data
            L2_distance = L2_distance * acc_sign
            total_L2_distance += L2_distance.sum()

            with torch.no_grad():
                outputs = tar_net(adv_inputs_ori)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                att_sign = torch.where(predicted == labels, zeros, ones)
                att_sign = att_sign + acc_sign
                att_sign = torch.where(att_sign == 2, ones, zeros)
                att_num += att_sign.sum().float()

    # Print final results
    print('Attack success rate: %.2f %%' %
          (att_num / acc_num * 100.0))
    print('l2 distance:  %.4f ' %
          (total_L2_distance / acc_num))


# ----------------------------
# Load Models
# ----------------------------

# Load pretrained target model (Net_m)
target_net = Net_m().to(device)
state_dict = torch.load('pretrained/net_m.pth')
target_net.load_state_dict(state_dict)
target_net = nn.DataParallel(target_net)
target_net.eval()

# Load attack model based on mode
if opt.mode == 'black':
    attack_net = Net_l().to(device)
    state_dict = torch.load('pretrained/net_l.pth')
    attack_net.load_state_dict(state_dict)
elif opt.mode == 'white':
    attack_net = target_net
elif opt.mode == 'dast':
    attack_net = Net_l().to(device)
    state_dict = torch.load('saved_model/netD_epoch_848.pth')  # Load trained DaST model
    attack_net.load_state_dict(state_dict)
    attack_net = nn.DataParallel(attack_net)

# ----------------------------
# Run Adversarial Evaluation
# ----------------------------
test_adver(attack_net, target_net, opt.adv, opt.target)
