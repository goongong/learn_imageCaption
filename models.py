import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models


class Encoder(nn.Modul):

    def __init__(self, embed_size, pretrained):
        super(Encoder, self).__init__()
        self.cnn = model.vgg16(pretrained=pretrained)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        for params in self.cnn.parameters():
            params.requires_grad = False
        self.embed_size = embed_size
        self.init_weights()

    def init_weights(self):
        self.cnn.fc.weights.data.normal_(0.0, 0.02)
        self.cnn.fc.bias.data.fill_(0)

    def forward(self, imgs, pretrained):
        features = self.vgg(imgs)
        return features


def Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(embed_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embedder = nn.embedder(vocab_size, embed_size)
        self.init_weights()

    def init_weights(self):
        self.embedder.weights.data.uniform_(-0.1, 0.1)
        self.linear.weights.data.uniform_(0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        embeddings = self.embedder(captions[:-1])
        packed_embeddings = torch.cat([features, embeddings], 0)
        packed_embeddings = pack_padded_sequence(pack_padded_sequence, lengths)
        out, _ = self.rnn(pack_padded_sequence)
        out = self.linear(out[0])
        return out
