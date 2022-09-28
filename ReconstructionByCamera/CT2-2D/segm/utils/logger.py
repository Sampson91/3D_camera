"""
https://github.com/facebookresearch/deit/blob/main/utils.py
"""

import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as distribution

import segm.utils.torch as ptu


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, format=None):
        if format is None:
            format = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.format = format

    def update(self, value, number=1):
        self.deque.append(value)
        self.count += number
        self.total += value * number

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        torch_tensor = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device=ptu.device
        )
        distribution.barrier()
        distribution.all_reduce(torch_tensor)
        torch_tensor = torch_tensor.tolist()
        self.count = int(torch_tensor[0])
        self.total = torch_tensor[1]

    @property
    def median(self):
        torch_median = torch.tensor(list(self.deque))
        return torch_median.median().item()

    @property
    def average(self):
        torch_mean = torch.tensor(list(self.deque), dtype=torch.float32)
        return torch_mean.mean().item()

    @property
    def global_average(self):
        return self.total / self.count

    @property
    def maximum(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.format.format(
            median=self.median,
            avg=self.average,
            global_avg=self.global_average,
            max=self.maximum,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.meters[key].update(value, n)

    def __getattr__(self, attribute):
        if attribute in self.meters:
            return self.meters[attribute]
        if attribute in self.__dict__:
            return self.__dict__[attribute]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attribute)
        )

    def __str__(self):
        loss_string = []
        for name, meter in self.meters.items():
            loss_string.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_string)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_frequency, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iteration_time = SmoothedValue(format="{avg:.4f}")
        data_time = SmoothedValue(format="{avg:.4f}")
        space_formation = ":" + str(len(str(len(iterable)))) + "d"
        log_message = [
            header,
            "[{0" + space_formation + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_message.append("max mem: {memory:.0f}")
        log_message = self.delimiter.join(log_message)
        memory_unit_Mb = 1024.0 * 1024.0
        for objective in iterable:
            data_time.update(time.time() - end)
            yield objective
            iteration_time.update(time.time() - end)
            if i % print_frequency == 0 or i == len(iterable) - 1:
                estimation_seconds = iteration_time.global_average * (len(iterable) - i)
                estimation_string = str(datetime.timedelta(seconds=int(estimation_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_message.format(
                            i,
                            len(iterable),
                            eta=estimation_string,
                            meters=str(self),
                            time=str(iteration_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / memory_unit_Mb,
                        ),
                        flush=True,
                    )
                    # self.update(iter_num=i)
                else:
                    print(
                        log_message.format(
                            i,
                            len(iterable),
                            eta=estimation_string,
                            meters=str(self),
                            time=str(iteration_time),
                            data=str(data_time),
                        ),
                        flush=True,
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_string = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_string, total_time / len(iterable)
            )
        )


def is_dist_avail_and_initialized():
    if not distribution.is_available():
        return False
    if not distribution.is_initialized():
        return False
    return True
