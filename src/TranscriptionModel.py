import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class TranscriptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, nhid, nlayers, max_length, num_classes):
        super(TranscriptionModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embed_dim, max_length)
        encoder_layers = TransformerEncoderLayer(embed_dim, nhead, nhid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, num_classes)
        
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self.generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Define the model
vocab_size = 100  # Set this to your vocabulary size
embed_dim = 512  # Embedding size for each token
nhead = 8        # Number of attention heads
nhid = 2048      # Hidden layer size in feed forward network inside transformer
nlayers = 4      # Number of Transformer blocks
max_length = 100 # Maximum sequence length
num_classes = vocab_size  # The output is also a sequence of tokens

model = TranscriptionModel(vocab_size, embed_dim, nhead, nhid, nlayers, max_length, num_classes)

# Example forward pass with random data
src = torch.rand(64, max_length).long()  # (batch size, sequence length)
out = model(src)
print(out.shape)  # Should be (batch size, sequence length, vocab size)
