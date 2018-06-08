# knn.py
# Author: Keanu Spies
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./images_small', help='path to dataset')
parser.add_argument('--netG', default='', help="path to netG")
opt = parser.parse_args()
print(opt)

dataset = dset.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)

ngpu = 0
nz = 100
ngf = 64
ndf = 64
nc = 3
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Reload Generator
device = torch.device("cpu")
netG = Generator(0).to(device)
netG.load_state_dict(torch.load(opt.netG, map_location='cpu'))

fixed_noise = torch.randn(64, 100, 1, 1)

# Loads data, creates generated artwork, performs knn
def knn(cuda=False):
	gdata = torch.utils.data.TensorDataset(netG(fixed_noise))
	gloader = torch.utils.data.DataLoader(gdata, batch_size=1)
	total = 0
	count = 0
	for i, data in enumerate(gloader, 0):
		return calculateDistance(data[0])

def calculateDistance(data):
	smallest_dist = float("inf")
	for i, datad in enumerate(dataloader, 0):
		distance = torch.sum(torch.norm(datad[0] - data))
		if distance < smallest_dist:
			smallest_dist = distance
	return smallest_dist

print(knn())
