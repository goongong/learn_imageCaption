class captionData(object):

    def __init__():

    def


def get_loader(batch_size, shuffle, sampler, num_workers, collate_fn):
    # get the dataset we want

    # interlize th dataset into batch_size
    data_loader = torch.utils.data.DataLoader(captionsData,
                                              batch_size,
                                              shuffle=shuffle,
                                              num_workers,
                                              collate_fn)
