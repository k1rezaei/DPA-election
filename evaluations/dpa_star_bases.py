from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from tqdm import tqdm
import json

INF = 10 ** 9

parser = argparse.ArgumentParser(description='Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations directory')
parser.add_argument('--vs', required=True, type=int, help='version of base classifiers')
parser.add_argument('--vr', required=True, type=int, help='version of base classifiers')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

k = 50
num_classes = 10

sum_of_logits = torch.zeros(10000, 50 ,num_classes).cuda()
first_it = True

prv_labels = None

for version in range(args.vs, args.vs + args.vr):
    filein = torch.load(args.evaluations + '_v' + str(version) + '.pth', map_location=torch.device(device))

    labels = filein['labels']
    scores = filein['scores']
    
    sum_of_logits += scores
    
    
    if not first_it:
        print(prv_labels == labels)
        
    prv_labels = labels
    first_it = False
    
    
sum_of_logits /= args.vr

print(sum_of_logits.shape)

torch.save({
    "scores": sum_of_logits,
    "labels": labels},
    'dpa_star_cifar_nin_baseline_FiniteAggregation_k50_d1.pth')
