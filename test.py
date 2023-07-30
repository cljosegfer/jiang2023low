import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np

from models import TextClassificationModel, zip_ssl
from datasets import AGNEWS
from utils import train, eval, pearson, acc, nomean, pearson_delta

EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 2048
EMBED_DIM = 64

import torch

from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import gzip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AGNEWS():
    def __init__(self):
        self.train_iter, self.test_iter = AG_NEWS()
        self.tokenizer = get_tokenizer("basic_english")

        self.vocab = build_vocab_from_iterator(self._yield_tokens(self.train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: int(x) - 1
    
    def _yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def loader(self, BATCH_SIZE, ssl = False):
        train_dataset = to_map_style_dataset(self.train_iter)
        test_dataset = to_map_style_dataset(self.test_iter)

        num_train = int(len(train_dataset) * 0.95)
        split_train_, split_valid_ = random_split(
            train_dataset, [num_train, len(train_dataset) - num_train]
        )

        if ssl:
            train_dataloader = DataLoader(split_train_, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_sll)
            valid_dataloader = DataLoader(split_valid_, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_sll)
            test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_sll)
        else:
            train_dataloader = DataLoader(split_train_, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_batch)
            valid_dataloader = DataLoader(split_valid_, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_batch)
            test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader

    def _collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    def _collate_sll(self, batch):
        raw_list, text_list, offsets, label_list = [], [], [0], []
        for _label, _text in batch:
            raw_list.append(_text)
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
            label_list.append(self.label_pipeline(_label))
        btsz = len(raw_list) // 2
        x_i = raw_list[:btsz]
        x_j = raw_list[btsz:]
        #ncd_list = [self._ncd(x_i[i], x_j[i]) for i in range(btsz)]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return raw_list, text_list.to(device), offsets.to(device), torch.tensor(label_list).to(device)
        #ncd = torch.tensor(ncd_list).to(device)
        #return ncd / torch.max(ncd), text_list.to(device), offsets.to(device)
    
    def _ncd(self, i, j):
        len_i_comp = len(gzip.compress(i.encode('utf-8')))
        len_j_comp = len(gzip.compress(j.encode('utf-8')))
        len_ij_comp = len(gzip.compress((" ".join([i,j])).encode('utf-8')))
        return len_ij_comp - min(len_i_comp, len_j_comp) / max(len_i_comp, len_j_comp)



ds = AGNEWS()

from sklearn.neighbors import KNeighborsClassifier
train_loader, val_loader, test_loader = ds.loader(BATCH_SIZE, ssl = True)
final_htrain = []
final_labels = []
y_train = []
for idx, (label, text, offsets, y) in enumerate(train_loader):
    final_labels.append(label)
    y_train.append(y.cpu().numpy())

#final_htrain = np.concatenate(final_htrain, axis=0)
final_labels = np.concatenate(final_labels, axis=0)
y_train = np.concatenate(y_train, axis=0)

from sklearn.neighbors import KNeighborsClassifier
train_loader, val_loader, test_loader = ds.loader(BATCH_SIZE, ssl = True)
final_labels_test = []
y_test = []
for idx, (label, text, offsets, y) in enumerate(test_loader):
    #print (label)
    final_labels_test.append(label)
    y_test.append(y.cpu().numpy())

#final_htrain = np.concatenate(final_htrain, axis=0)
final_labels_test = np.concatenate(final_labels_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

#calculate ncd of final_labels and calculate knn, as described in the algorithm from the paper
k=2
final_yhat = []
for x1 in tqdm(final_labels_test):
    c_x1 = len(gzip.compress(x1.encode('utf-8')))
    distance_from_x1 = []
    for x2 in (final_labels):
        c_x2 = len(gzip.compress(x2.encode('utf-8')))
        c_x1_x2 = len(gzip.compress((x1+x2).encode('utf-8')))
        distance_from_x1.append((c_x1_x2 - min(c_x1, c_x2)) / max(c_x1, c_x2))

    #calculate knn
    sorted_idx = np.argsort(distance_from_x1)
    topk_class = y_train[sorted_idx[:k]]
    yhat = np.argmax(np.bincount(topk_class))
    final_yhat.append(yhat)

#calculate accuracy
final_yhat = np.array(final_yhat)
from sklearn.metrics import accuracy_score

print("Accuracy: ", accuracy_score(y_test, final_yhat))
    

            
