import numpy as np
from torch.utils import data


def InfiniteSampler(number):
    # i = 0
    i = number - 1
    order = np.random.permutation(number)
    while True:
        yield order[i]
        i += 1
        if i >= number:
            np.random.seed()
            order = np.random.permutation(number)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.number_of_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.number_of_samples))

    def __len__(self):
        return 2 ** 31
