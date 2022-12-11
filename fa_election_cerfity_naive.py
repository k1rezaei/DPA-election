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
import random
from tqdm import tqdm

INF = int(10 ** 9)
frac1, frac2, frac3 = 0, 0, 0

def get_num_of_poisoned_sample(gap, gap_reducers):
    gap_reducers = gap_reducers.sort(descending=True)[0]
    num_of_poisoned_samples = 0
    while gap - gap_reducers[num_of_poisoned_samples] >= 0:
        gap -= gap_reducers[num_of_poisoned_samples].item()
        num_of_poisoned_samples += 1

    return num_of_poisoned_samples

parser = argparse.ArgumentParser(description='Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations file')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--k', default = 50, type=int, help='number of partitions')
parser.add_argument('--d', default = 1, type=int, help='number of partitions that each model is trained on')


args = parser.parse_args()

args.n_subsets = args.k * args.d

random.seed(999999999+208)
shifts = random.sample(range(args.n_subsets), args.d)


if not os.path.exists('./radii'):
    os.makedirs('./radii')
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

filein = torch.load('/cmlscratch/wwx/DPA/evaluations/'+args.evaluations + '.pth', map_location=torch.device(device))
labels = filein['labels']
scores = filein['scores']

num_classes = args.num_classes
num_of_points = scores.shape[0]
num_models = scores.shape[1]

max_classes = torch.argsort(scores, dim=2, descending=True)
idx_election = torch.zeros((num_of_points, ), dtype=torch.int)

predictions = torch.zeros(num_of_points, num_classes)
for i in range(num_models):
	predictions[(torch.arange(num_of_points),max_classes[:,i, 0])] += 1

certs = torch.LongTensor(num_of_points)

#prepared for indexing
shifted = [
    [(h + shift)%args.n_subsets for shift in shifts] for h in range(args.n_subsets)
]
shifted = torch.LongTensor(shifted)

for i in tqdm(range(num_of_points)):

    prediction = predictions[i].cpu().numpy()

    ordered_classes = np.argsort(-prediction, kind='stable')
    m1 = ordered_classes[0].item()
    m2 = ordered_classes[1].item()

    # election
    m1_election = np.zeros(num_classes)
    m2_election = np.zeros(num_classes)

    for cls in range(num_classes):
        m1_election[cls] = 2 * (scores[i, :, m1] > scores[i, :, cls]).sum().item() - num_models
        m2_election[cls] = 2 * (scores[i, :, m2] > scores[i, :, cls]).sum().item() - num_models

    
        
    elec = m1_election[m2]
    if elec > 0:
        idx_election[i] = m1
    elif elec == 0:
        if m1 <= m2:
            idx_election[i] = m1
        else:
            idx_election[i] = m2
    else:
        idx_election[i] = m2


    if idx_election[i] != labels[i]:
        certs[i] = -1
        continue
    
    
    certs[i] = args.n_subsets #init value
    label = int(labels[i])

    max_classes_given_h = max_classes[i, shifted.view(-1), 0].view(-1, args.d)
    m1_election_given_h = torch.zeros((num_classes, args.d * args.k, args.d,))
    m2_election_given_h = torch.zeros((num_classes, args.d * args.k, args.d,))

    # print(max_classes_given_h.shape)

    for cls in range(num_classes):
        m1_election_given_h[cls] = (scores[i, shifted.view(-1), m1] > scores[i, shifted.view(-1), cls]).view(-1, args.d)
        m2_election_given_h[cls] = (scores[i, shifted.view(-1), m2] > scores[i, shifted.view(-1), cls]).view(-1, args.d)
    
    case1, case2, case3 = INF, INF, INF

    # case1: (top two classes remain same)
    case1_gap = 0
    
    if elec > 0:
        case1_gap = (elec - (m2 <= m1))
        deltas_case1 = 2 * m1_election_given_h[m2].sum(dim=1)
        case1 = get_num_of_poisoned_sample(case1_gap, deltas_case1)
    elif elec == 0:
        case1_gap = 0
        case1 = 0
    else:
        case1_gap = (-elec - (m1 <= m2))
        deltas_case1 = 2 * m2_election_given_h[m1].sum(dim=1)
        case1 = get_num_of_poisoned_sample(case1_gap, deltas_case1)

    # case2: (we want to keep the prediction, and change the other class to sth else)
    case2 = INF
    if idx_election[i] == m1:
        for m3 in range(num_classes):
            if m1 == m3 or m2 == m3:
                continue

            case21_gap = prediction[m2] - prediction[m3] - (m3 <= m2)
            deltas_case21 = (1 + (max_classes_given_h == m2).long() - (max_classes_given_h == m3).long()).sum(dim=1)
            n1 = get_num_of_poisoned_sample(case21_gap, deltas_case21)

            case22_gap = m1_election[m3] - (m3 <= m1)
            deltas_case22 = 2 * m1_election_given_h[m3].sum(dim=1)
            n2 = get_num_of_poisoned_sample(case22_gap, deltas_case22)

            m3_need = max(n1, n2)
            case2 = min(case2, m3_need)
    else:
        for m3 in range(num_classes):
            if m1 == m3 or m2 == m3:
                continue
            case21_gap = prediction[m1] - prediction[m3] - (m3 <= m1)
            deltas_case21 = (1 + (max_classes_given_h == m1).long() - (max_classes_given_h == m3).long()).sum(dim=1)
            n1 = get_num_of_poisoned_sample(case21_gap, deltas_case21)

            case22_gap = m2_election[m3] - (m3 <= m2)
            deltas_case22 = 2 * m2_election_given_h[m3].sum(dim=1)
            n2 = get_num_of_poisoned_sample(case22_gap, deltas_case22)

            m3_need = max(n1, n2)
            case2 = min(case2, m3_need)
    
    # case3: (we want to take out the prediction)
    case3 = INF
    if idx_election[i] == m1:
        for m3 in range(num_classes):
            if m1 == m3 or m2 == m3:
                continue
                
            case31_gap = prediction[m1] - prediction[m3] - (m3 <= m1)
            deltas_case31 = (1 + (max_classes_given_h == m1).long() - (max_classes_given_h == m3).long()).sum(dim=1)
            n1 = get_num_of_poisoned_sample(case31_gap, deltas_case31)
            # TODO: how to come-up with a better upper-bound? this one is a bit loose.

            case3 = min(case3, n1)
    else:
        for m3 in range(num_classes):
            if m1 == m3 or m2 == m3:
                continue
            
            case32_gap = prediction[m2] - prediction[m3] - (m3 <= m2)
            deltas_case32 = (1 + (max_classes_given_h == m2).long() - (max_classes_given_h == m3).long()).sum(dim=1)
            n1 = get_num_of_poisoned_sample(case32_gap, deltas_case32)
            case3 = min(case3, n1)

    certs[i] = min(min(case1, case2), case3)
    
    # check that adversary takes which of these cases
    if case1 == certs[i]:
        frac1 += 1
    elif case2 == certs[i]:
        frac2 += 1
    elif case3 == certs[i]:
        frac3 += 1
    

s = frac1 + frac2 + frac3
print(s)
print(frac1/s, frac2/s, frac3/s)

base_acc = 100 *  (max_classes[:, :, 0] == labels.unsqueeze(1)).sum().item() / (num_of_points * num_models)
print('Base classifier accuracy: ' + str(base_acc))
torch.save(certs,'./radii/fa_election_'+args.evaluations+'.pth')
a = certs.cpu().sort()[0].numpy()
accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/predictions.shape[0]
print('Smoothed classifier accuracy: ' + str(accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(accs >= .5)))
print(accs)
print('==================')


