import rlbench.utils as rlbench_utils
from helpers import utils
from helpers.utils import create_obs_config
from os.path import join, exists
import os
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
from agents import peract_bc
import torch.distributed as dist

from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['MASTER_ADDR'] = "localhost"
os.environ['MASTER_PORT'] = "29500"

CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
REWARD_SCALE = 100.0
LOW_DIM_SIZE = 4
dist.init_process_group("gloo",
                        rank=0,
                        world_size=1)

path = 'data'
ts = 50
replay = peract_bc.launch_utils.create_recon_replay(1,1,CAMERAS)
peract_bc.launch_utils.fill_recon_replay(replay, path, CAMERAS, ts, 0, 10)
# print(replay.get_storage_signature())
