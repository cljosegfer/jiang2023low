
import torch
from torch.nn import functional as F

import gzip

def train(model, loader, criterion, optimizer, ssl = False):
    model.train()
    cost = 0
    for idx, (label, text, offsets) in enumerate(loader):
        if ssl:
            btsz = len(offsets) // 2
            h = model.forward(text, offsets)
            delta = F.pairwise_distance(h[:btsz], h[btsz:])
            loss = criterion(delta, label)
        else:
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        cost += loss.item()
    return cost / loader.dataset.__len__()

def eval(model, loader, criterion, ssl = False):
    model.eval()
    cost = 0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(loader):
            if ssl:
                btsz = len(offsets) // 2
                h = model.forward(text, offsets)
                delta = F.pairwise_distance(h[:btsz], h[btsz:])
                loss = criterion(delta, label)
            else:
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
            cost += loss.item()
    return cost / loader.dataset.__len__()

def acc(model, loader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(loader):
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def pearson(hi, hj, ncd):
    delta = F.pairwise_distance(hi, hj)

    vx = delta - torch.mean(delta)
    vy = ncd - torch.mean(ncd)

    return -F.cosine_similarity(vx, vy, dim = 0)

def nomean(hi, hj, ncd):
    delta = F.pairwise_distance(hi, hj)

    # vx = delta - torch.mean(delta)
    # vy = ncd - torch.mean(ncd)

    # return -F.cosine_similarity(vx, vy, dim = 0)
    return -F.cosine_similarity(delta, ncd, dim = 0)

def pearson_delta(delta, ncd):
    # delta = F.pairwise_distance(hi, hj)

    vx = delta - torch.mean(delta)
    vy = ncd - torch.mean(ncd)

    return -F.cosine_similarity(vx, vy, dim = 0)