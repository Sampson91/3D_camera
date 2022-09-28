import os
import torch
import torch.distributed

"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(mode, local_rank):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    gpu_id = int(os.environ["LOCAL_RANK"])

    dist_rank = 0
    # world_size = len(os.environ.get('CUDA_VISIBLE_DEVICES').split(','))
    world_size = torch.cuda.device_count()
    distributed = world_size > 1
    use_gpu = mode

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
