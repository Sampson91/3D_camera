import numpy as np
from torch.utils import data


def InfiniteSampler(number):
    # i = 0
    iteration = number - 1
    order = np.random.permutation(number)
    while True:
        yield order[iteration]
        iteration += 1
        if iteration >= number:
            np.random.seed()
            order = np.random.permutation(number)
            iteration = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
