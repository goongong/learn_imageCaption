import pickle
import argparse
from collections import Counter
import json
import os
import random


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json_file, threshold):
    """Build a simple vocabulary wrapper."""
    data = json.load(open(json_file, 'r'))
    data = data['images']
    counter = Counter()
    ids = [i for i in range(len(data))]
    for i, id in enumerate(ids):
        caption = random.choice(data[id]['sentences'])['tokens']
        counter.update(caption)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." % (i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is
    # discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(json_file=args.json_file,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Total vocabulary size: %d" % len(vocab))
    print("Saved the vocabulary wrapper to '%s'" % vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str,
                        default='./data/flickr8k/dataset.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/flickr8k/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
