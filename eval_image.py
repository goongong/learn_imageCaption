import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from models import Encoder, Decoder
from PIL import Image
import json
import scipy.io
import random


def main(args):
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    datasets = json.load(open(args.json_file, 'rb'))['images'][7000:7000 + 128]
    images_feats = scipy.io.loadmat(args.mat_file)['feats'].transpose()[
        7000:7000 + 128]
    images_feats = torch.Tensor(images_feats)
    captions = []
    for index in range(len(datasets)):
        sentence = random.choice(datasets[index]['sentences'])['raw']
        captions.append(sentence)

    # Build Models
    decoder = Decoder(args.in_features, len(vocab),
                      args.embed_size, args.hidden_size)

    # Load the trained model parameters
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare Image
    # image = Image.open(args.image)
    # image_tensor = Variable(transform(image).unsqueeze(0))

    # Set initial states
    state = (Variable(torch.zeros(args.num_layers, 1, args.hidden_size)),
             Variable(torch.zeros(args.num_layers, 1, args.hidden_size)))

    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        state = [s.cuda() for s in state]
        image_tensor = image_tensor.cuda()

    # Generate caption from image
    feature = Variable(images_feats)
    sampled_ids = decoder.sample(feature, state)
    sampled_ids = sampled_ids.cpu().data.numpy()
    exit(1)
    # Decode word_ids to words
    predict_sentences = []
    for sentence_id in sampled_ids:
        sampled_caption = []
        for word_id in sentence_id:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
        predict_sentence = ' '.join(sampled_caption)
        predict_sentences.append(predict_sentence)
    for truth, predict in zip(captions, predict_sentences):
        print '{}=>{}'.format(truth, predict)
    # Print out image and generated caption.
    # print (sentence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', type=str, required=True,
    #                     help='input image for generating caption')
    parser.add_argument('--json_file', type=str, default='./data/flickr8k/dataset.json',
                        help='the test json file contains the captions')
    parser.add_argument('--mat_file', type=str, default='./data/flickr8k/vgg_feats.mat',
                        help='the image features file')
    # parser.add_argument('--encoder_path', type=str, default='./models/encoder-5-3000.pkl',
    #                     help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-5-41.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/flickr8k/vocab.pkl',
                        help='path for vocabulary wrapper')
    # parser.add_argument('--crop_size', type=int, default=224,
    #                     help='size for center cropping images')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--in_features', type=int, default=4096,
                        help='the size of input features')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
