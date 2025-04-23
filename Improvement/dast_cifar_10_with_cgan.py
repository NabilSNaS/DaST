# Enable compatibility with Python 2 print syntax
from __future__ import print_function
# Command-line argument parsing
import argparse
import os
import math
import gc
import sys
import xlwt
import random
import numpy as np
# from advertorch.attacks import LinfBasicIterativeAttack, L2BasicIterativeAttack
import foolbox as fb
from foolbox.criteria import Misclassification, TargetedMisclassification
# from advertorch.attacks import L2PGDAttack
import joblib
# from utils import load_data
# import pickle
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
# from net import Net_s, Net_m, Net_l
from torchvision.models import resnet18, ResNet18_Weights
from vgg import VGG
from resnet import ResNet18
from foolbox.attacks import L2CarliniWagnerAttack
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F


cudnn.benchmark = True
# workbook = xlwt.Workbook(encoding = 'utf-8')
# worksheet = workbook.add_sheet('imitation_network_sig')
nz = 128
target =False
# Set CUDA GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 1000
# Ensure reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
	    pass

# Log outputs to file
sys.stdout = Logger('dast_cifar10.log', sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
#parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--beta', type=float, default=0.1, help='beta')#(from 0.1 to 20.0)--DasTP     0.0--DasTL
parser.add_argument('--G_type', type=int, default=1, help='G type')
parser.add_argument('--save_folder', type=str, default='saved_model', help='alpha')
opt = parser.parse_args()
print(opt)


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transforms = transforms.Compose([transforms.ToTensor()])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transforms)

# Initialize substitute model (netD)
netD = VGG('VGG13').cuda()

# Initialize target (oracle) model
original_net = VGG('VGG16').cuda()
# original_net = nn.DataParallel(original_net)
# Load pre-trained weights for target model
original_net.load_state_dict(torch.load(
        '/work/pi_csc592_uri_edu/hafija_uri/DAST/vgg_vgg16_final.pth'))
# original_net = nn.DataParallel(original_net)
# Set target model to evaluation mode
original_net.eval()


# Wrap netD with Foolbox for adversarial attacks
fmodel = fb.PyTorchModel(netD, bounds=(0.0,1.0))
# BIM
# Define the L2 BIM adversarial attack
attack_fb = fb.attacks.L2BasicIterativeAttack(abs_stepsize=0.01, steps=240, random_start=False)
# # FGSM
# attack_fb = fb.attacks.FGSM()
# PGD
# attack_fb = fb.attacks.LinfPGD(steps=40, abs_stepsize=0.01, random_start=True)
# C&W
# attack_fb = L2CarliniWagnerAttack()
# attack_fb = L2CarliniWagnerAttack(
#     binary_search_steps=5,
#     steps=500,
#     confidence=10,
#     initial_const=0.1,
# )





nc=3

data_list = [i for i in range(6000, 8000)] # fast validation
# Load test samples for validation
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=2)
# nc=1

device = torch.device("cuda:0" if opt.cuda else "cpu")
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Custom loss for imitation and diversity
class Loss_max(nn.Module):
    def __init__(self):
        super(Loss_max, self).__init__()
        return

    def forward(self, pred, truth, proba):
        criterion_1 = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        pred_prob = F.softmax(pred, dim=1)
        loss = criterion(pred, truth) + criterion_1(pred_prob, proba) * opt.beta
        # loss = criterion(pred, truth)
        final_loss = torch.exp(loss * -1)
        return final_loss

# Evaluate model against adversarial attacks
def get_att_results(model, target):
    correct = 0.0
    total = 0.0
    total_L2_distance = 0.0
    att_num = 0.
    acc_num = 0.
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        if target:
            # randomly choose the specific label of targeted attack
            labels = torch.randint(0, 9, (inputs.size(0),)).to(device)
            # test the images which are not classified as the specific label

            ones = torch.ones_like(predicted)
            zeros = torch.zeros_like(predicted)
            acc_sign = torch.where(predicted == labels, zeros, ones)
            acc_num += acc_sign.sum().float()
            # adv_inputs_ori = adversary.perturb(inputs, labels)
            
#             For PGD Epsilon 0.06, BIM 1.5 and for FGSM 0.03
             # _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, TargetedMisclassification(labels), epsilons=0.06)
            
            _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, TargetedMisclassification(labels), epsilons=0.8)
            # _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, TargetedMisclassification(labels), epsilons=0.03)
            
            # C&W
            # adv_inputs_ori = attack_fb.run(fmodel, inputs, TargetedMisclassification(labels.to(device)))

            
            L2_distance = (adv_inputs_ori - inputs).squeeze()
            # L2_distance = (torch.linalg.norm(L2_distance, dim=list(range(1, inputs.squeeze().dim())))).data
            L2_distance = (torch.linalg.norm(L2_distance.flatten(start_dim=1), dim=1)).data
            # L2_distance = (torch.linalg.matrix_norm(L2_distance, dim=0, keepdim=True)).data
            L2_distance = L2_distance * acc_sign
            total_L2_distance += L2_distance.sum()
            with torch.no_grad():
                outputs = model(adv_inputs_ori)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum()
                att_sign = torch.where(predicted == labels, ones, zeros)
                att_sign = att_sign + acc_sign
                att_sign = torch.where(att_sign == 2, ones, zeros)
                att_num += att_sign.sum().float()
        else:
            ones = torch.ones_like(predicted)
            zeros = torch.zeros_like(predicted)
            acc_sign = torch.where(predicted == labels, ones, zeros)
            acc_num += acc_sign.sum().float()
            # adv_inputs_ori = adversary.perturb(inputs, labels)
            _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, Misclassification(labels.to(device)), epsilons=0.8)
            # _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, Misclassification(labels.to(device)), epsilons=0.06)
            # _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, Misclassification(labels.to(device)), epsilons=0.03)
            
            #C&W

            # adv_inputs_ori = attack_fb.run(fmodel, inputs, Misclassification(labels.to(device)))
            #Non-targeted FGSM attack
            # adv_inputs_ori = attack_fb.run(fmodel, inputs, labels, epsilon=0.01)
            
            
            L2_distance = (adv_inputs_ori - inputs).squeeze()
            L2_distance = (torch.linalg.norm(L2_distance.flatten(start_dim=1), dim=1)).data
            # L2_distance = (torch.linalg.matrix_norm(L2_distance, dim=0, keepdim=True)).data
            L2_distance = L2_distance * acc_sign
            total_L2_distance += L2_distance.sum()
            with torch.no_grad():
                outputs = model(adv_inputs_ori)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum()
                att_sign = torch.where(predicted == labels, zeros, ones)
                att_sign = att_sign + acc_sign
                att_sign = torch.where(att_sign == 2, ones, zeros)
                att_num += att_sign.sum().float()

    if target:
        att_result = (att_num / acc_num * 100.0)
        # print('Attack success rate: %.2f %%' %
        #       ((att_num / acc_num * 100.0)))
    else:
        att_result = (att_num / acc_num * 100.0)
        # print('Attack success rate: %.2f %%' %
        #       (att_num / acc_num * 100.0))
    print('l2 distance:  %.4f ' % (total_L2_distance / acc_num))
    return att_result


# Preprocessing network for generator noise input
class pre_conv(nn.Module):
    def __init__(self, num_class):
        super(pre_conv, self).__init__()
        self.nf = 64
        if opt.G_type == 1:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                # nn.LeakyReLU(0.2, inplace=True)
                nn.ReLU(True),
            )
        elif opt.G_type == 2:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),  # added

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.ReLU(True),

                nn.Conv2d(self.nf, self.shape[0], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.shape[0]),
                nn.ReLU(True),

                nn.Conv2d(self.shape[0], self.shape[0], 3, 1, 1, bias=False),
                # if self.shape[0] == 3:
                #     nn.Tanh()
                # else:
                nn.Sigmoid()
            )
    def forward(self, input):
        output = self.pre_conv(input)
        return output

pre_conv_block = []
for i in range (10):
    # pre_conv_block.append(nn.DataParallel(pre_conv(10).cuda()))
    pre_conv_block.append(pre_conv(10).cuda())

class Generator(nn.Module):
    def __init__(self, num_class):
        super(Generator, self).__init__()
        self.nf = 64
        self.num_class = num_class
        if opt.G_type == 1:
            self.main = nn.Sequential(
                nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 4),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf, nc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(nc),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
                nn.Sigmoid()
            )
        elif opt.G_type == 2:
            self.main = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 8, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True)
            )
    def forward(self, input):
        output = self.main(input)
        return output


# Generator model for CIFAR-10 image synthesis

class Generator_cifar10(nn.Module):
    def __init__(self, num_class):
        super(Generator_cifar10, self).__init__()
        self.nf = 64
        self.num_class = num_class
        self.noise_dim = nz  # nz is the noise vector length (e.g. 128)
        self.embed_dim = 128  # dimension for label embeddings
        # Label embedding layer to produce a vector for each class
        self.label_emb = nn.Embedding(num_class, self.embed_dim)
        # Total input to generator is noise + label embedding
        in_channels = self.noise_dim + self.embed_dim
        # Define DCGAN-like generator architecture
        self.main = nn.Sequential(
            # 1. First transpose conv: from 1x1 to 4x4
            nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            # 2. Upsample to 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            # 3. Upsample to 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            # 4. Upsample to 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            # 5. Final conv to get 3 channels (32x32, no change in spatial size)
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()  # output in [0,1] range for CIFAR images
        )

    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        label_vec = self.label_emb(labels)               # shape: (B, embed_dim)
        label_vec = label_vec.unsqueeze(2).unsqueeze(3)  # shape: (B, embed_dim, 1, 1)
        noise = noise.view(noise.size(0), -1, 1, 1)      # ensure noise is (B, noise_dim, 1, 1)
        gen_input = torch.cat((noise, label_vec), dim=1) # (B, noise_dim+embed_dim, 1, 1)
        return self.main(gen_input)

def chunks(arr, m):
    n = int(math.ceil(arr.size(0) / float(m)))
    return [arr[i:i + n] for i in range(0, arr.size(0), n)]

# Initialize generator network
netG = Generator_cifar10(10).cuda()
# netG.apply(weights_init)
# netG = nn.DataParallel(netG)

criterion = nn.CrossEntropyLoss()
criterion_max = Loss_max()

# setup optimizer
# Optimizer for substitute model (netD)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerD =  optim.SGD(netD.parameters(), lr=opt.lr*20.0, momentum=0.0, weight_decay=5e-4)
# Optimizer for generator (netG)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr*2.0, betas=(opt.beta1, 0.999))
# optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr*100.0, weight_decay=5e-4)
# optimizerG =  optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer_block = []
for i in range(10):
    optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

with torch.no_grad():
    correct_netD = 0.0
    total = 0.0
    netD.eval()
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        # print(inputs.size())
        labels = labels.cuda()
        # outputs = netD(inputs)
        outputs = original_net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_netD += (predicted == labels).sum()
    print('original net accuracy: %.2f %%' %
            (100. * correct_netD.float() / total))
# Non-targeted Pretrained
# Evaluate adversarial success rate
att_result = get_att_results(original_net, target=False)

# Dast - P 
# Evaluate adversarial success rate
# att_result = get_att_results(netD, target=False)

print('Attack success rate: %.2f %%' %
        (att_result))

batch_num = 500
best_accuracy = 0.0
best_att = 0.0
cnt =0

# Main training loop
# Initialize networks and optimizers
# netD = VGG('VGG13').cuda()
# original_net = VGG('VGG16').cuda()
# original_net.load_state_dict(torch.load(
#         '/work/pi_csc592_uri_edu/hafija_uri/DAST/vgg_vgg16_final.pth'))
# original_net.eval()  # target model

netG = Generator_cifar10(num_class=10).cuda()
netG.apply(weights_init)  # apply DCGAN weight init
criterion = nn.CrossEntropyLoss()
criterion_max = Loss_max()
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr * 2.0, betas=(opt.beta1, 0.999))

# Training loop
num_classes = 10
batch_num = 500  # number of batches per epoch (as in original code)
for epoch in range(opt.niter):
    netD.train()
    for ii in range(batch_num):
        # (1) Update D network (surrogate model)
        netD.zero_grad()
        # Sample noise and labels for a full batch
        noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
        if opt.batchSize % num_classes == 0:
            # evenly distribute labels 0-9
            labels = torch.arange(num_classes, device=device).repeat_interleave(opt.batchSize // num_classes)
        else:
            labels = torch.randint(0, num_classes, (opt.batchSize,), device=device)
        # Generate synthetic CIFAR-10 images conditioned on these labels
        fake_images = netG(noise, labels)
        # Shuffle the batch (mix class order)
        perm = torch.randperm(fake_images.size(0))
        fake_images = fake_images[perm]; labels = labels[perm]
        
        #save_fake image
        os.makedirs("generated_samples", exist_ok=True)

        # Save first 10 generated images individually
        for i in range(10):
            img = fake_images[i]
            label = labels[i].item()
            vutils.save_image(img, f"generated_samples/sample_{i}_label_{label}.png")

        # Query the black-box (target) model for outputs on synthetic data
        with torch.no_grad():
            teacher_logits = original_net(fake_images)
        teacher_prob = F.softmax(teacher_logits, dim=1)              # target model predicted probabilities
        teacher_pred = teacher_logits.max(1)[1]                      # target model predicted label indices

        # Surrogate model forward on synthetic data
        output_D = netD(fake_images.detach())                        # .detach(): don't backprop into G here
        pred_prob_D = F.softmax(output_D, dim=1)
        # Surrogate loss: imitate target (cross-entropy + MSE to probabilities)
        lossD_cls = criterion(output_D, teacher_pred)                # match target model's predicted class
        lossD_prob = mse_loss(pred_prob_D, teacher_prob)             # match full probability distribution
        errD = lossD_cls + opt.beta * lossD_prob
        errD.backward()
        if errD.item() > 0.3:                                        # update D only if loss is above threshold
            optimizerD.step()

        # (2) Update G network (generator)
        netG.zero_grad()
        # Forward through surrogate again (now with grad) to evaluate generator
        output_D_for_G = netD(fake_images)                           # no .detach(), allow grad to flow into G
        # Imitation loss: encourage netD's output to match target model (uses Loss_max)
        loss_imitate = criterion_max(pred=output_D_for_G, truth=teacher_pred, proba=teacher_prob)
        # Label consistency (diversity) loss: encourage netD to predict the conditioned label
        loss_diverse = criterion(output_D_for_G, labels)
        # Combined generator loss
        errG = opt.alpha * loss_diverse + loss_imitate
        errG.backward()
        optimizerG.step()




    netD.eval()
    
    # Nnon -targeted Pretrained
# Evaluate adversarial success rate
    # att_result = get_att_results(original_net, target=False)
    #Nno-targeted Dast-P
# Evaluate adversarial success rate
    att_result = get_att_results(netD, target=False)

    print('Attack success rate: %.2f %%' % (att_result))

    if best_att < att_result:
# Save trained substitute model
        torch.save(netD.state_dict(),
                        opt.save_folder+'/netD_epoch_%d.pth' % (epoch))
        torch.save(netG.state_dict(),
                        opt.save_folder+'/netG_epoch_%d.pth' % (epoch))
        best_att = att_result
        print('model saved')
    else:
        print('Best ASR: %.2f %%' % (best_att))

    ################################################
    # evaluate the accuracy of trained D:
    ################################################
    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        netD.eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = netD(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
        print('substitute accuracy: %.2f %%' %
                (100. * correct_netD.float() / total))
        if best_accuracy < correct_netD:
# Save trained substitute model
            torch.save(netD.state_dict(),
                       opt.save_folder+'/netD_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder+'/netG_epoch_%d.pth' % (epoch))
            best_accuracy = correct_netD
            print('model saved')
        else:
            print('Best ACC: %.2f %%' % (100. * best_accuracy.float() / total))
#     worksheet.write(epoch, 1, (correct_netD.float() / total).item())
# workbook.save('imitation_network_saved_cifar10.xls')
