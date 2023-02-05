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
import gc
import time

gc.enable()
st = time.time()

INF = int(10 ** 9)
frac1, frac2, frac3 = 0, 0, 0

# CertV1 
def get_sample_cert(gap, gap_reducers):
    gap_reducers = gap_reducers.sort(descending=True)[0]
    sample_cert = 0
    while gap > 0:
        gap -= gap_reducers[sample_cert].item()
        sample_cert += 1

    return sample_cert

parser = argparse.ArgumentParser(description='Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations file')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--k', default = 50, type=int, help='number of partitions')
parser.add_argument('--d', default = 1, type=int, help='number of partitions that each model is trained on')


args = parser.parse_args()

args.n_subsets = args.k * args.d

random.seed(999999999+208)
shifts = random.sample(range(args.n_subsets), args.d)


if not os.path.exists('./certs'):
    os.makedirs('./certs')

device = 'cpu'

filein = torch.load('evaluations/'+args.evaluations + '.pth', map_location=torch.device(device))
labels = filein['labels']
scores = filein['scores']


num_of_classes = args.num_classes
num_of_samples = scores.shape[0]
num_of_models = scores.shape[1]

max_classes = torch.argsort(scores, dim=2, descending=True)
idx_roe_fa = torch.zeros((num_of_samples, ), dtype=torch.int) # roe+fa prediction

predictions = torch.zeros(num_of_samples, num_of_classes) # number of first-round votes for each class
for i in range(num_of_models):
	predictions[(torch.arange(num_of_samples),max_classes[:,i, 0])] += 1

certs = torch.LongTensor(num_of_samples)

#prepared for indexing
shifted = [
    [(h + shift)%args.n_subsets for shift in shifts] for h in range(args.n_subsets)
]
shifted = torch.LongTensor(shifted)

for i in range(num_of_samples):
    # FA+ROE
    
    # votes in 1st round
    prediction = predictions[i].cpu().numpy()
    ordered_classes = np.argsort(-prediction, kind='stable')

    # top two classes
    m1 = ordered_classes[0].item()
    m2 = ordered_classes[1].item()

    # votes in 2nd round
    m1_election = np.zeros(num_of_classes)
    m2_election = np.zeros(num_of_classes)

    for cls in range(num_of_classes):
        m1_election[cls] = 2 * (scores[i, :, m1] > scores[i, :, cls]).sum().item() - num_of_models
        m2_election[cls] = 2 * (scores[i, :, m2] > scores[i, :, cls]).sum().item() - num_of_models
    
    # FA+ROE prediction
    elec = m1_election[m2]
    if elec > 0:
        idx_roe_fa[i] = m1
    elif elec == 0: # tie
        if m1 <= m2:
            idx_roe_fa[i] = m1
        else:
            idx_roe_fa[i] = m2
    else:
        idx_roe_fa[i] = m2


    if idx_roe_fa[i] != labels[i]: # wrong prediction
        certs[i] = -1
        continue
    
    c_pred = idx_roe_fa[i]
    c_sec = m1 + m2 - c_pred
    
    certs[i] = args.n_subsets #init value
    label = int(labels[i])

    max_classes_given_h = max_classes[i, shifted.view(-1), 0].view(-1, args.d)
    m1_election_given_h = torch.zeros((num_of_classes, args.d * args.k, args.d,))
    m2_election_given_h = torch.zeros((num_of_classes, args.d * args.k, args.d,))

    m1_to_m3 = np.zeros(num_of_classes)
    m2_to_m3 = np.zeros(num_of_classes)

    # pw_{b, m1, m3} and pw_{b, m2, m3} in Round 1
    for m3 in range(num_of_classes):
        gap = prediction[m1] - prediction[m3] + (m3 > m1) # this gap should become non-positive
        pw = (1 + (max_classes_given_h == m1).long() - (max_classes_given_h == m3).long()).sum(dim=1) # how much each partition can contribute to reduce gap
        m1_to_m3[m3] = get_sample_cert(gap, pw) # greedy approach

        gap = prediction[m2] - prediction[m3] + (m3 > m2) # this gap should become non-positive
        pw = (1 + (max_classes_given_h == m2).long() - (max_classes_given_h == m3).long()).sum(dim=1) # how much each partition can contribute to reduce gap
        m2_to_m3[m3] = get_sample_cert(gap, pw) # greedy approach
    
    # pw_{b, m1, cls} in Round 2
    for cls in range(num_of_classes):
        m1_election_given_h[cls] = (scores[i, shifted.view(-1), m1] > scores[i, shifted.view(-1), cls]).view(-1, args.d)
        m2_election_given_h[cls] = (scores[i, shifted.view(-1), m2] > scores[i, shifted.view(-1), cls]).view(-1, args.d)
    
    
    if c_pred == m1:
        R1 = m1_to_m3
        R1_csec = m2_to_m3
        R2_gaps = m1_election
        R2_pw = m1_election_given_h
    else:
        R1 = m2_to_m3
        R1_csec = m1_to_m3
        R2_gaps = m2_election
        R2_pw = m2_election_given_h
    
    # CertR1 = min Certv2(c_pred, c1, c2)
    CertR1 = INF
    
    for c1 in range(num_of_classes):
        if c1 == c_pred:
            continue
        
        for c2 in range(c1):
            if c2 == c_pred or c2 == c1:
                continue
            
            n1 = R1[c1]
            n2 = R1[c2]

            gap = prediction[c_pred] - prediction[c1] + (c1 > c_pred) + prediction[c_pred] - prediction[c2] + (c2 > c_pred)
            pw = (1 + 2 * (max_classes_given_h == c_pred).long() - (max_classes_given_h == c1).long() - (max_classes_given_h == c2).long()).sum(dim=1)
            nsum = get_sample_cert(gap, pw)
            
            Certv2_c1_c2 = max(max(n1, n2), nsum)
            CertR1 = min(CertR1, Certv2_c1_c2)
    
    # CertR2 = min max{Certv1({f_i}, c_sec, c), Certv1({g_i}, c_pred, c)}
    CertR2 = INF
    
    for c in range(num_of_classes):
        if c == c_pred:
            continue
        
        CertR2_c_1 = R1_csec[c]
        gap = R2_gaps[c] + (c > c_pred)
        pw = 2 * R2_pw[c].sum(dim=1)
        CertR2_c_2 = get_sample_cert(gap, pw)

        CertR2_c = max(CertR2_c_1, CertR2_c_2)
        CertR2 = min(CertR2, CertR2_c)

    assert(CertR1 > 0 and CertR2 > 0)
    certs[i] = min(CertR1, CertR2) - 1

    if i % 1000 == 0:
        print(f'{i} / {num_of_samples} ...')
    

en = time.time()
print(en - st)
print('done ...')


base_acc = 100 *  (max_classes[:, :, 0] == labels.unsqueeze(1)).sum().item() / (num_of_samples * num_of_models)
print('Base classifier accuracy: ' + str(base_acc))
torch.save(certs,'./certs/fa_roe_'+args.evaluations+'.pth')
a = certs.cpu().sort()[0].numpy()
accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/predictions.shape[0]
print('Smoothed classifier accuracy: ' + str(accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(accs >= .5)))
print(accs)
print('==================')
