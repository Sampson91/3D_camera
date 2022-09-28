from timm import scheduler
# from timm import optim
from torch import optim as optim

from segm.optim.scheduler import PolynomialLR
import torch.nn as nn


def create_scheduler(opt_args, optimizer):
    if opt_args.sched == "polynomial":
        learning_rate_scheduler = PolynomialLR(
            optimizer,
            opt_args.poly_step_size,
            opt_args.iter_warmup,
            opt_args.iter_max,
            opt_args.poly_power,
            opt_args.min_learning_rate,
        )
    else:
        learning_rate_scheduler, _ = scheduler.create_scheduler(opt_args, optimizer)
    return learning_rate_scheduler

def get_parameter_groups(model, weight_decay=0):
    parameter_list = []
    for name, parameter_ in model.named_parameters():
        if not parameter_.requires_grad:
            continue
        parameter_list.append(parameter_)
    return parameter_list


def create_optimizer(args, model, filter_bias_and_bn=True):
    optimizer_lower = args.optimizer.lower()
    weight_decay = args.weight_decay        # 0
    if filter_bias_and_bn:
        print('finetune last 6 blocks in vit')
        parameters = get_parameter_groups(model, weight_decay)
    else:
        parameters = model.parameters()
        print('finetune all vit blocks..')
    if optimizer_lower == 'sgd' or optimizer_lower == 'nesterov':
        optimizer = optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=weight_decay)
        #  SGD, lr optim 内部的
    else:
        assert False and "Invalid optimizer"
        raise ValueError
    return optimizer


