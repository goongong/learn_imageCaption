import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, Decoder
import torch.optim as optim
import argparse
import os
import cPickle as pickle
from build_vocab import Vocabulary
from data_loader import get_loader
import numpy as np


def main(args):
    # load datasets
    vocab = pickle.load(open(args.vocab_path, 'rb'))
    train_data = get_loader(args.json_file,
                            args.mat_file,
                            vocab,
                            args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers)
    # get vocab
    # build model
    if args.encoder:
        encoder = Encoder(args.embed_size, True)

    decoder = Decoder(args.in_features, len(vocab),
                      args.embed_size, args.hidden_size)
    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())
    if args.encoder:
        params = list(decoder.parameters()) + list(encoder.cnn.fc.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)
    # train
    total_step = len(train_data)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(train_data):
            if args.encoder:
                images_features = encoder(Variable(images))
            else:
                images_features = Variable(images)
            captions = Variable(captions)
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]

            decoder.zero_grad()
            outputs = decoder(images_features, captions, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % args.disp_step == 0:
                print('Epoch [%d/%d], step [%d/%d], loss: %.4f, Perplexity: %5.4f'
                      % (epoch, args.num_epochs, i, total_step, loss.data[0], np.exp(loss.data[0])))

            # Save the models
            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='./data/flickr8k/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--json_file', type=str, default='./data/flickr8k/dataset.json',
                        help='the caption file')
    parser.add_argument('--mat_file', type=str, default='./data/flickr8k/vgg_feats.mat',
                        help='the image features file')
    parser.add_argument('--disp_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')
    parser.add_argument('--encoder', type=int, default=0,
                        help='use the encoder or not')
    # Model parameters
    parser.add_argument('--in_features', type=int, default=4096,
                        help='the size of image features')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
