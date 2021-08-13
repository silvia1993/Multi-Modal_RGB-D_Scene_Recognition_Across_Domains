import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def accuracy(output, target, topk=(1,)):
    
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def mean_acc(target_indice, pred_indice, num_classes, classes=None):
    
    assert(num_classes == len(classes))

    acc = 0.
    
    print('{0} Class Acc Report {1}'.format('#' * 10, '#' * 10))
    
    for i in range(num_classes):
        idx = np.where(target_indice == i)[0]
        class_correct = accuracy_score(target_indice[idx], pred_indice[idx])
        acc += class_correct
        print('acc {0}: {1:.3f}'.format(classes[i], class_correct * 100))
    
    print('#' * 30)
    
    return (acc / num_classes) * 100

def process_output(output):
    # Computes the result and argmax index
    pred, index = output.topk(1, 1, largest=True)
    return pred.cpu().float().numpy().flatten(), index.cpu().numpy().flatten()
