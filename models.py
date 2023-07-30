
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc = nn.Sequential(nn.Linear(embed_dim, 128),
            nn.Mish(),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 512),
            nn.Mish(),
            nn.Linear(512, 1024),
            nn.Mish(),
            nn.Linear(1024, 2048),
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.fc.weight.data.uniform_(-initrange, initrange)
        #self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.fc(embedded)
        return embedded


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_heads, num_encoder_layers):
        super(TransformerClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, src):
        # src: (seq_len, batch_size)
        embedded = self.embedding(src) * math.sqrt(self.embed_dim)
        encoded = self.transformer_encoder(embedded)
        pooled = F.avg_pool1d(encoded.transpose(0, 1), encoded.size(0)).squeeze(2)
        out = F.relu(self.fc(pooled))
        out = self.fc2(out)
        return out
