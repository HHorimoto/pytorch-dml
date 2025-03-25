import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence(logits_1, logits_2):
    softmax_1 = F.softmax(logits_1, dim=1)
    softmax_2 = F.softmax(logits_2, dim=1)
    kl = (softmax_2 * torch.log((softmax_2 / (softmax_1+1e-10)) + 1e-10)).sum(dim=1)
    return kl.mean()