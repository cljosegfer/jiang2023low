
import torch

from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import gzip
import numpy as np

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
        self.split_train_, self.split_valid_ = random_split(
            train_dataset, [num_train, len(train_dataset) - num_train]
        )

        if ssl:
            train_dataloader = DataLoader(self.split_train_, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_sll)
            valid_dataloader = DataLoader(self.split_valid_, batch_size = BATCH_SIZE, shuffle = False, collate_fn = self._collate_sll)
            test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = self._collate_sll)
        else:
            train_dataloader = DataLoader(self.split_train_, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_batch)
            valid_dataloader = DataLoader(self.split_valid_, batch_size = BATCH_SIZE, shuffle = False, collate_fn = self._collate_batch)
            test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = self._collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader
    
    def repair(self, BATCH_SIZE):
        np.random.shuffle(self.split_train_.indices)
        train_dataloader = DataLoader(self.split_train_, batch_size = BATCH_SIZE, shuffle = True, collate_fn = self._collate_sll)

        return train_dataloader

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
        ncd_list = [self._ncd(x_i[i], x_j[i]) for i in range(btsz)]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        ncd = torch.tensor(ncd_list).to(device)
        #return ncd / torch.max(ncd), text_list.to(device), offsets.to(device)
        return raw_list, text_list.to(device), offsets.to(device), torch.tensor(label_list).to(device), ncd / 1000
    
    def _ncd(self, i, j):
        len_i_comp = len(gzip.compress(i.encode('utf-8')))
        len_j_comp = len(gzip.compress(j.encode('utf-8')))
        len_ij_comp = len(gzip.compress((" ".join([i,j])).encode('utf-8')))
        return len_ij_comp - min(len_i_comp, len_j_comp) / max(len_i_comp, len_j_comp)



class AGNEWSTransformer():
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