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

def ceil(a: int, b: int): # ceil(max(a, 0) / b)
    return (max(0, a) + b - 1) // b

def load_json():
    with open('../data/array_v2.json', 'r') as f:
        return np.array(json.loads(f.read()), dtype=int)

dp_array = load_json()

def dp(g1, g2):
    return dp_array[g1][g2]

INF = 10 ** 9

parser = argparse.ArgumentParser(description='Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations directory')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print(device)


filein = torch.load(args.evaluations+'.pth', map_location=torch.device(device))

labels = filein['labels']
scores = filein['scores']

max_classes = torch.argsort(scores, dim=2, descending=True).cpu()

num_of_classes = args.num_classes
num_of_samples = scores.shape[0]
num_of_models = scores.shape[1]

cert_dpa = torch.zeros((num_of_samples, ), dtype=torch.int) # cert of samples using DPA
cert_dpa_roe = torch.zeros((num_of_samples, ), dtype=torch.int) # cert of samples using DPA+ROE

idx_dpa = torch.zeros((num_of_samples, ), dtype=torch.int).cuda() # DPA prediction
idx_dpa_roe = torch.zeros((num_of_samples, ), dtype=torch.int).cuda() # DPA+ROE prediction

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

    # DPA
    # DPA prediction
    idx_dpa[i] = m1
    
    # DPA Cert
    gap = prediction[m1] - prediction[m2] + (m2 > m1)
    assert(gap > 0)
    cert_dpa[i] = ceil(gap, 2) - 1
    
    
    # DPA+ROE
    m1_election = np.zeros(num_of_classes, dtype=int)
    m2_election = np.zeros(num_of_classes, dtype=int)

    for cls in range(num_of_classes):
        m1_election[cls] = 2 * (scores[i, :, m1] > scores[i, :, cls]).sum().item() - num_of_models
        m2_election[cls] = 2 * (scores[i, :, m2] > scores[i, :, cls]).sum().item() - num_of_models
    
    # DPA+ROE prediction
    elec = m1_election[m2]
    if elec > 0:
        idx_dpa_roe[i] = m1
    elif elec == 0: # tie
        if m1 <= m2:
            idx_dpa_roe[i] = m1
        else:
            idx_dpa_roe[i] = m2
    else:
        idx_dpa_roe[i] = m2
        
    c_pred = idx_dpa_roe[i]
    c_sec = m1 + m2 - c_pred
    
    if c_pred == m1:
        R2 = m1_election
    else:
        R2 = m2_election

    # DPA+ROE Certificate min(CertR1, CertR2)
    
    # CertR1 = min Certv2(c_pred, c1, c2)
    CertR1 = INF
    
    for c1 in range(num_of_classes):
        if c1 == c_pred:
            continue
        
        for c2 in range(num_of_classes):
            if c2 == c_pred or c2 == c1:
                continue
                
            gap1 = prediction[c_pred] - prediction[c1] + (c1 > c_pred)
            gap2 = prediction[c_pred] - prediction[c2] + (c2 > c_pred)
            
            gap1 = max(gap1, 0)
            gap2 = max(gap2, 0)
            
            assert(dp(gap1, gap2) > 0)

            CertR1 = min(CertR1, dp(gap1, gap2))
            
    
    # CertR2 = min max{Certv1({f_i}, c_sec, c), Certv1({g_i}, c_pred, c)}
    CertR2 = INF
    
    for c in range(num_of_classes):
        if c == c_pred:
            continue
        
        g = prediction[c_sec] - prediction[c] + (c > c_sec)
        CertR2_c_1 = ceil(g, 2)
        
        g = R2[c] + (c > c_pred)
        CertR2_c_2 = ceil(g, 2)
        
        CertR2_c = max(CertR2_c_1, CertR2_c_2)
        CertR2 = min(CertR2, CertR2_c)
    
    assert(CertR1 > 0 and CertR2 > 0)
    
    cert_dpa_roe[i] = min(CertR1, CertR2) - 1

B = np.array([5, 10, 15, 18, 20])

print("==> original DPA ..")
certs = cert_dpa
torchidx = idx_dpa
certs[torchidx != labels] = -1
torch.save(certs,'../certs/dpa_star_'+args.evaluations+ '.pth')
a = certs.cpu().sort()[0].numpy()

dpa_accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/num_of_samples
print('Smoothed classifier accuracy: ' + str(dpa_accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(dpa_accs >= .5)))
print(dpa_accs)
print(np.round_(100.* dpa_accs[B], 2))

print('==================')


print("==> DPA+ROE ..")
certs = cert_dpa_roe
torchidx = idx_dpa_roe
certs[torchidx != labels] = -1
torch.save(certs,'../certs/dpa_star_roe_'+args.evaluations+ '.pth')

a = certs.cpu().sort()[0].numpy()

roe_dpa_accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/num_of_samples
print('Smoothed classifier accuracy: ' + str(roe_dpa_accs[0] * 100.) + '%')
print('Robustness certificate: ' + str(sum(roe_dpa_accs >= .5)))
print(roe_dpa_accs)
print(np.round_(100.* roe_dpa_accs[B], 2))
print('==================')

print(np.round_(100.* (roe_dpa_accs[B] - dpa_accs[B]), 2))
