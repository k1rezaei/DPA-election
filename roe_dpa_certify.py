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

def load_json():
    with open('data/array.json', 'r') as f:
        return np.array(json.loads(f.read()), dtype=int)

INF = 10 ** 9

parser = argparse.ArgumentParser(description='Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations directory')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')

args = parser.parse_args()
if not os.path.exists('./certs'):
    os.makedirs('./certs')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

dp_array = load_json()

filein = torch.load('/cmlscratch/wwx/DPA/evaluations/'+args.evaluations + '.pth', map_location=torch.device(device))

labels = filein['labels']
scores = filein['scores']

max_classes = torch.argsort(scores, dim=2, descending=True)

num_of_classes = args.num_classes
num_of_samples = scores.shape[0]
num_of_models = scores.shape[1]

rad_dpa = torch.zeros((num_of_samples, ), dtype=torch.int) # radius of samples using dpa
rad_roe_dpa = torch.zeros((num_of_samples, ), dtype=torch.int) # radius of samples using roe + dpa

idx_dpa = torch.zeros((num_of_samples, ), dtype=torch.int).cuda() # dpa prediction
idx_roe_dpa = torch.zeros((num_of_samples, ), dtype=torch.int).cuda() # roe+dpa prediction

predictions = torch.zeros(num_of_samples, num_of_classes, dtype=int) # number of first-round votes for each class
for i in range(num_of_models):
	predictions[(torch.arange(num_of_samples), max_classes[:, i, 0])] += 1

for i in tqdm(range(num_of_samples)):
    # votes in 1st round
    prediction = predictions[i].cpu().numpy()
    ordered_classes = np.argsort(-prediction, kind='stable') 

    # top two classes
    m1 = ordered_classes[0].item()
    m2 = ordered_classes[1].item()

    # dpa
    gap = prediction[m1] - prediction[m2] - (m2 <= m1)
    assert(gap >= 0)

    rad_dpa[i] = int(gap / 2)
    idx_dpa[i] = m1

    # roe + dpa
    m1_election = np.zeros(num_of_classes, dtype=int)
    m2_election = np.zeros(num_of_classes, dtype=int)

    for cls in range(num_of_classes):
        m1_election[cls] = 2 * (scores[i, :, m1] > scores[i, :, cls]).sum().item() - num_of_models
        #number of models which prefer m1 over {cls} minus number of models which prefer {cls} over m1

        m2_election[cls] = 2 * (scores[i, :, m2] > scores[i, :, cls]).sum().item() - num_of_models
        #number of models which prefer m2 over {cls} minus number of models which prefer {cls} over m2
    
    # roe + dpa prediction
    elec = m1_election[m2]
    if elec > 0:
        idx_roe_dpa[i] = m1
    elif elec == 0: # tie
        if m1 <= m2:
            idx_roe_dpa[i] = m1
        else:
            idx_roe_dpa[i] = m2
    else:
        idx_roe_dpa[i] = m2

    # case1: (eliminating prediction in the 1st round)
    case1 = INF
    if idx_roe_dpa[i] == m1: # m1 is prediction
        for m3 in range(num_of_classes):
            if m1 == m3 or m2 == m3:
                continue
            
            n2 = prediction[m1] - prediction[m2] - (m2 <= m1) # this gap should become negative
            n3 = prediction[m1] - prediction[m3] - (m3 <= m1) # this gap should become negative
            assert(n2 >= 0 and n3 >= 0)

            m3_need = dp_array[n2][n3] # preprocessed dp array.
            case1 = min(case1, m3_need)

    else: # m2 is prediction
        for m3 in range(num_of_classes):
            if m1 == m3 or m2 == m3:
                continue

            n1 = prediction[m2] - prediction[m3] - (m3 <= m2) # this gap should become negative
            assert(n1 >= 0)

            m3_need = n1 // 2
            case1 = min(case1, m3_need)

    
    # case2: (eliminating prediction in the 2nd round)
    #### sub-case: (not changing two classes of 1st round)
    case21 = 0
    if elec > 0:
        case21 = (elec - (m2 <= m1))
    elif elec == 0:
        case21 = 0
    else:
        case21 = (-elec - (m1 <= m2))

    assert(case21 >= 0)
    case21 = case21 // 2
    
    #### sub-case: (changing two classes of 1st round)
    case22 = INF
    if idx_roe_dpa[i] == m1: # m1 is prediction
        for m3 in range(num_of_classes):
            if m1 == m3 or m2 == m3:
                continue
            n1 = prediction[m2] - prediction[m3] - (m3 <= m2) # this gap should become negative
            n2 = m1_election[m3] - (m3 <= m1) # this gap should become negative
            assert(n1 >= 0)

            m3_need = max(n1, n2) // 2
            case22 = min(case22, m3_need)
    else: # m2 is prediction
        for m3 in range(num_of_classes):
            if m1 == m3 or m2 == m3:
                continue
            n1 = prediction[m1] - prediction[m3] - (m3 <= m1) # this gap should become negative
            n2 = m2_election[m3] - (m3 <= m2) # this gap should become negative
            assert(n1 >= 0)

            m3_need = max(n1, n2) // 2
            case22 = min(case22, m3_need)

    
    rad_roe_dpa[i] = min(case21, min(case22, case1))

print("==> original dpa ..")
certs = rad_dpa
torchidx = idx_dpa
certs[torchidx != labels] = -1
torch.save(certs,'./certs/dpa_'+args.evaluations+'.pth')
a = certs.cpu().sort()[0].numpy()

dpa_accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/num_of_samples
print('Smoothed classifier accuracy: ' + str(dpa_accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(dpa_accs >= .5)))
print(dpa_accs)
print('==================')


print("==> election dpa ..")
certs = rad_roe_dpa
torchidx = idx_roe_dpa
certs[torchidx != labels] = -1
torch.save(certs,'./certs/roe_dpa_'+args.evaluations+'.pth')
a = certs.cpu().sort()[0].numpy()

roe_dpa_accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/num_of_samples
print('Smoothed classifier accuracy: ' + str(roe_dpa_accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(roe_dpa_accs >= .5)))
print(roe_dpa_accs)
print('==================')

