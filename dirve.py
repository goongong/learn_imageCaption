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


def main(args):
    # load datasets
    vocab = pickle.load(open(args.vocab_path, 'rb'))
    train_data = get_loader(args.image_dir, args.json_dir, vocab, transform,
                            args.batch_size, args.num_workers)
    # get vocab
    # build model
    encoder = Encoder(args.embed_size, True)
    decoder = Decoder(len(vocab), args.embed_size, args.hidden_size)
    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.cnn.fc.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate)
    # train
    total_step = len(train_data)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) enumerate(train_data):
            images = Variable(images)
            captions = Variable(captions)
            targets = pack_padded_sequence(captions, lengths)[0]
            images_features = encoder(images)
            outputs = decoder(images_features, captions, lengths)

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            if i % arg.disp_step == 0:
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


if __name__ == '__mian__':
    parser = argparse.ArgumentParser()
    parser.add_argument('')
    args = parser.parse_args()
    main(args)
