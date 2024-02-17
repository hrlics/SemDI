import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def negative_sampling(example):
    if example['labels'] == 0:
        return random.random() > 0.7
    return True 

def compute_metrics(gold, predicted):
    c_predict = 0 
    c_correct = 0
    c_gold = 0

    for g, p in zip(gold, predicted):
        if g != 0:
            c_gold += 1
        if p != 0:
            c_predict += 1
        if g != 0 and p != 0:
            c_correct += 1

    p = c_correct / (c_predict + 1e-100)
    r = c_correct / c_gold
    f = 2 * p * r / (p + r + 1e-100)
    
    return p, r, f

def record_best_scores(timestamp, precision, recall, f1, filename):
    with open(filename, 'a') as file:
        file.write(f"{timestamp}\t{precision*100:.2f}\t{recall*100:.2f}\t{f1*100:.2f}\n")