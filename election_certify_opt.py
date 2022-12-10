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
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')

args = parser.parse_args()
if not os.path.exists('./radii'):
    os.makedirs('./radii')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

filein = torch.load('/cmlscratch/wwx/DPA/evaluations/'+args.evaluations + '.pth', map_location=torch.device(device))
# cifar_nin_baseline_OneInMany_k50_d1.pth

labels = filein['labels'] #(10,000)
scores = filein['scores'] #(10,000, 50, 10)

print(scores.shape)
print(labels.shape)


max_classes = torch.argsort(scores, dim=2, descending=True) #(10,000, 50, 10)
# sort is stable? TODO

num_classes = args.num_classes
num_of_points = scores.shape[0]
num_models = scores.shape[1]

diffs_dpa = torch.zeros((num_of_points, ), dtype=torch.int)
diffs_election_dpa = torch.zeros((num_of_points, ), dtype=torch.int)
idx_dpa = torch.zeros((num_of_points, ), dtype=torch.int)
idx_election = torch.zeros((num_of_points, ), dtype=torch.int)

predictions = torch.zeros(num_of_points, num_classes)
for i in range(num_models):
	predictions[(torch.arange(num_of_points),max_classes[:,i, 0])] += 1

for i in tqdm(range(num_of_points)):
    # top two votes ...
    prediction = predictions[i].cpu().numpy()

    ordered_classes = np.argsort(-prediction, kind='stable')
    m1 = ordered_classes[0].item()
    m2 = ordered_classes[1].item()

    # dpa
    gap = prediction[m1] - prediction[m2] - (m2 <= m1)
    assert(gap >= 0) # check if sort is stable

    diffs_dpa[i] = int(gap / 2)
    idx_dpa[i] = m1


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

    # case1: (top two classes remain same).
    case1 = 0
    if elec > 0:
        case1 = (elec - (m2 <= m1))
    elif elec == 0:
        case1 = 0
    else:
        case1 = (-elec - (m1 <= m2))

    # case2: (we want to keep the prediction, and change the other class to sth else).
    case2 = INF
    if idx_election[i] == m1:
        for m3 in range(num_classes):
            if m1 == m3 or m2 == m3:
                continue
            n1 = prediction[m2].item() - prediction[m3] - (m3 <= m2)
            n2 = m1_election[m3] - (m3 <= m1)

            m3_need = max(n1, n2)
            case2 = min(case2, m3_need)
    else:
        for m3 in range(num_classes):
            if m1 == m3 or m2 == m3:
                continue
            n1 = prediction[m1] - prediction[m3] - (m3 <= m1)
            n2 = m2_election[m3] - (m3 <= m2)
            m3_need = max(n1, n2)
            case2 = min(case2, m3_need)

    # case3: (we want to take out the prediction)
    case3 = INF
    if idx_election[i] == m1:
        for m3 in range(num_classes):
            if m1 == m3 or m2 == m3:
                continue
                
            n1 = prediction[m1] - prediction[m3] - (m3 <= m1)
            new_m1 = prediction[m1] - int(n1/2) - 1
            
            n2 = new_m1 - prediction[m2] - (m2 <= m1)
            if n2 < 0:
                m3_need = n1
            else:
                m3_need = 2 * (int(n1/2) + int(n2/2) + 1)
            case3 = min(case3, m3_need)
    else:
        for m3 in range(num_classes):
            if m1 == m3 or m2 == m3:
                continue

            n1 = prediction[m2] - prediction[m3] - (m3 <= m2)
            case3 = min(case3, n1)

    diffs_election_dpa[i] = int(min(case1, min(case2, case3)) / 2)

print("==> original dpa ..")
certs = diffs_dpa
torchidx = idx_dpa
certs[torchidx != labels] = -1
torch.save(certs,'./radii/dpa_'+args.evaluations+'_v2.pth')
a = certs.cpu().sort()[0].numpy()

dpa_accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/num_of_points
print('Smoothed classifier accuracy: ' + str(dpa_accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(dpa_accs >= .5)))
print(dpa_accs)
print('==================')


print("==> election dpa ..")
certs = diffs_election_dpa
torchidx = idx_election
certs[torchidx != labels] = -1
torch.save(certs,'./radii/election_'+args.evaluations+'_v2.pth')
a = certs.cpu().sort()[0].numpy()

election_accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/num_of_points
print('Smoothed classifier accuracy: ' + str(election_accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(election_accs >= .5)))
print(election_accs)
print('==================')

