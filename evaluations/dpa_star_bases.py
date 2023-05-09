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
parser.add_argument('--vr', required=True, type=int, help='num of versions of base classifiers')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

k = 100
num_classes = 43

sum_of_logits = None
first_it = True

prv_labels = None

# versions = [i for i in range(1, 11)]
# versions.extend([12, 13, 16, 17, 21, 22])
# print(versions)

for version in tqdm(range(args.vs, args.vs + args.vr)):
# for version in tqdm(versions):
    filein = torch.load(args.evaluations + '_v' + str(version) + '.pth', map_location=torch.device(device))

    labels = filein['labels']
    scores = filein['scores']
    
    if sum_of_logits is None:
        sum_of_logits = torch.zeros(scores.shape[0], scores.shape[1], scores.shape[2]).cuda()
        
    sum_of_logits += scores
    
    
    prv_labels = labels
    first_it = False
    
    
# sum_of_logits /= args.vr
# sum_of_logits /= len(versions)

print(sum_of_logits.shape)

torch.save({
    "scores": sum_of_logits,
    "labels": labels},
    f'dpa_star_{args.evaluations[:-3]}_d{args.vr}.pth')
