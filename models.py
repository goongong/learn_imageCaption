import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models


class Encoder(nn.Module):

    def __init__(self, embed_size, pretrained):
        super(Encoder, self).__init__()
        self.cnn = model.vgg16(pretrained=pretrained)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        for params in self.cnn.parameters():
            params.requires_grad = False
        self.embed_size = embed_size
        self.init_weights()

    def init_weights(self):
        self.cnn.fc.weight.data.normal_(0.0, 0.02)
        self.cnn.fc.bias.data.fill_(0)

    def forward(self, imgs, pretrained):
        features = self.vgg(imgs)
        return features


class Decoder(nn.Module):

    def __init__(self, in_features, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embedder = nn.Embedding(vocab_size, embed_size)
        self.init_weights()

    def init_weights(self):
        self.embedder.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        img_features = self.fc(features).unsqueeze(1)
        # print(img_features.size())
        embeddings = self.embedder(captions)
        # print(embeddings.size())
        packed_embeddings = torch.cat([img_features, embeddings], 1)
        packed_embeddings = pack_padded_sequence(
            packed_embeddings, lengths, batch_first=True)
        out, _ = self.rnn(packed_embeddings)
        out = self.linear(out[0])
        return out
