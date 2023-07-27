
import torch
import torch.nn as nn

class TextClassificationModel(nn.Module):
    def __init__(self, embed_dim, vocab_size = 95811, num_class = 4):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

class zip_ssl(nn.Module):
    def __init__(self, embed_dim, vocab_size = 95811):
        super(zip_ssl, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse = False)
        # self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # self.fc.weight.data.uniform_(-initrange, initrange)
        # self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        # return self.fc(embedded)
        return embedded
