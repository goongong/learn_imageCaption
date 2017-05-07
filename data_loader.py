import json
import scipy.io
import torch
import torch.utils.data as data


class captionData(data.Dataset):

    def __init__(self, json_file, mat_file, vocab):
        self.datset = json.load(open('json_file', 'rb'))
        features_struct = scipy.io.loadmat(mat_file)
        self.feats = features_struct['feats']
        split = defaultdict(list)
        for img in self.dataset['images']:
            split[img['split']].append(img)
        self.imags = split['train']
        self.vocab = vocab

    def __getitem__(self, index):
        vacab = self.vocab
        image_feat = self.feats[index]
        image_feat = torch.Tensor(image_feat)
        caption = []
        caption.append(vocab('<start>'))
        sentence = random.choice(self.imags[index]['sentences'])
        caption.append(sentence)
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image_feat, caption

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(json_file, mat_file, vocab, batch_size, shuffle, num_workers):
    # get the dataset we want
    captionsData = captionsData(json_file, mat_file, vocab)
    # interlize th dataset into batch_size
    data_loader = torch.utils.data.DataLoader(captionsData,
                                              batch_size,
                                              shuffle=shuffle,
                                              num_workers,
                                              collate_fn)
