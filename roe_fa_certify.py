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

def get_sample_radius(gap, gap_reducers):
    gap_reducers = gap_reducers.sort(descending=True)[0]
    sample_rad = 0
    while gap - gap_reducers[sample_rad] >= 0:
        gap -= gap_reducers[sample_rad].item()
        sample_rad += 1

    return sample_rad

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

filein = torch.load('/cmlscratch/wwx/DPA/evaluations/'+args.evaluations + '.pth', map_location=torch.device(device))
labels = filein['labels']
scores = filein['scores']

print('loading done ...')

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

print('all memory is allocated ...')

for i in range(num_of_samples):
    # roe + fa

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
        #number of models which prefer m1 over {cls} minus number of models which prefer {cls} over m1

        m2_election[cls] = 2 * (scores[i, :, m2] > scores[i, :, cls]).sum().item() - num_of_models
        #number of models which prefer m2 over {cls} minus number of models which prefer {cls} over m2
    
    
    # roe + fa prediction
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
    
    
    certs[i] = args.n_subsets #init value
    label = int(labels[i])

    max_classes_given_h = max_classes[i, shifted.view(-1), 0].view(-1, args.d)
    m1_election_given_h = torch.zeros((num_of_classes, args.d * args.k, args.d,))
    m2_election_given_h = torch.zeros((num_of_classes, args.d * args.k, args.d,))

    m1_to_m3 = np.zeros(num_of_classes)
    m2_to_m3 = np.zeros(num_of_classes)

    # take m3 to be selected in the 1st round
    for m3 in range(num_of_classes):
        if m1 == m3 or m2 == m3:
            continue

        case_gap = prediction[m1] - prediction[m3] - (m3 <= m1) # this gap should become negative
        assert(case_gap >= 0)

        deltas_case = (1 + (max_classes_given_h == m1).long() - (max_classes_given_h == m3).long()).sum(dim=1) # how much each partition can contribute to reduce gap
        m1_to_m3[m3] = get_sample_radius(case_gap, deltas_case) # greedy approach

        case_gap = prediction[m2] - prediction[m3] - (m3 <= m2) # this gap should become negative
        assert(case_gap >= 0)
        deltas_case = (1 + (max_classes_given_h == m2).long() - (max_classes_given_h == m3).long()).sum(dim=1) # how much each partition can contribute to reduce gap
        m2_to_m3[m3] = get_sample_radius(case_gap, deltas_case) # greedy approach
    

    for cls in range(num_of_classes):
        m1_election_given_h[cls] = (scores[i, shifted.view(-1), m1] > scores[i, shifted.view(-1), cls]).view(-1, args.d)
        m2_election_given_h[cls] = (scores[i, shifted.view(-1), m2] > scores[i, shifted.view(-1), cls]).view(-1, args.d)
    
    case21, case22, case1 = INF, INF, INF

     # case1: (eliminating prediction in the 1st round)
    case1 = INF
    if idx_roe_fa[i] == m1: # m1 is prediction
        # OLD CASE
        # for m3 in range(num_of_classes):
        #     if m1 == m3 or m2 == m3:
        #         continue
                
        #     n1 = m1_to_m3[m3]

        #     case1_gap = prediction[m1] - prediction[m3] - (m3 <= m1) + prediction[m1] - prediction[m2] - (m2 <= m1)
        #     deltas_case1 = (1 + 2 * (max_classes_given_h == m1).long() - (max_classes_given_h == m2).long() - (max_classes_given_h == m3).long()).sum(dim=1)
        #     n2 = get_sample_radius(case1_gap, deltas_case1)

        #     m3_need = max(n1, n2)
        #     case1 = min(case1, m3_need)
        
        # NEW CASE
        for m3 in range(num_of_classes):
            if m1_to_m3[m3] > case1:
                continue

            for m4 in range(m3):
                if m3 == m1 or m4 == m1:
                    continue

                n1 = m1_to_m3[m3]
                n2 = m1_to_m3[m4]

                if n2 > case1:
                    continue    

                case1_gap = prediction[m1] - prediction[m3] - (m3 <= m1) + prediction[m1] - prediction[m4] - (m4 <= m1)
                deltas_case1 = (1 + 2 * (max_classes_given_h == m1).long() - (max_classes_given_h == m3).long() - (max_classes_given_h == m4).long()).sum(dim=1)
                n3 = get_sample_radius(case1_gap, deltas_case1)
                
                m34_need = max(max(n1, n2), n3)
                case1 = min(case1, m34_need)

    else: # m2 is prediction
        for m3 in range(num_of_classes):
            if m1 == m3 or m2 == m3:
                continue
            
            n1 = m2_to_m3[m3]
            case1 = min(case1, n1)

    # case2: (eliminating prediction in the 2nd round)
    #### sub-case: (not changing two classes of 1st round)
    case21_gap = 0
    
    if elec > 0:
        case21_gap = (elec - (m2 <= m1))
        deltas_case21 = 2 * m1_election_given_h[m2].sum(dim=1) # how much each partition can contribute to reduce gap
        case21 = get_sample_radius(case21_gap, deltas_case21) # greedy approach
    elif elec == 0:
        case21_gap = 0
        case21 = 0
    else:
        case21_gap = (-elec - (m1 <= m2))
        deltas_case21 = 2 * m2_election_given_h[m1].sum(dim=1) # how much each partition can contribute to reduce gap
        case21 = get_sample_radius(case21_gap, deltas_case21) # greedy approach

    #### sub-case: (changing two classes of 1st round)
    case22 = INF
    if idx_roe_fa[i] == m1: # m1 is prediction
        for m3 in range(num_of_classes):
            if m1 == m3 or m2 == m3:
                continue

            n1 = m2_to_m3[m3]

            case22_gap = m1_election[m3] - (m3 <= m1)
            deltas_case22 = 2 * m1_election_given_h[m3].sum(dim=1)
            n2 = get_sample_radius(case22_gap, deltas_case22)

            m3_need = max(n1, n2)
            case22 = min(case22, m3_need)
    else: # m2 is prediction
        for m3 in range(num_of_classes):
            if m1 == m3 or m2 == m3:
                continue
            
            n1 = m1_to_m3[m3]

            case22_gap = m2_election[m3] - (m3 <= m2)
            deltas_case22 = 2 * m2_election_given_h[m3].sum(dim=1)
            n2 = get_sample_radius(case22_gap, deltas_case22)

            m3_need = max(n1, n2)
            case22 = min(case22, m3_need)
    

    certs[i] = min(min(case21, case22), case1)

    if i % 1000 == 0:
        print(f'{i} / {num_of_samples} ...')
    

en = time.time()
print(en - st)
print('done ...')


base_acc = 100 *  (max_classes[:, :, 0] == labels.unsqueeze(1)).sum().item() / (num_of_samples * num_of_models)
print('Base classifier accuracy: ' + str(base_acc))
torch.save(certs,'./certs_v2/roe_fa_'+args.evaluations+'.pth')
a = certs.cpu().sort()[0].numpy()
accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/predictions.shape[0]
print('Smoothed classifier accuracy: ' + str(accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(accs >= .5)))
print(accs)
print('==================')
