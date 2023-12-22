import os
import copy
import pickle
import gc
import logging
from typing import List
import json

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from agents import c2farm_lingunet_bc
from agents import peract_bc
from agents import arm
from agents.baselines import bc_lang, vit_bc_lang


def run_seed(rank,
             cfg: DictConfig,
             obs_config: ObservationConfig,
             cams,
             multi_task,
             seed,
             world_size) -> None:

    world_size=1
    dist.init_process_group("gloo",
                            rank=rank,
                            world_size=world_size)
    
    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks

    task_folder = task if not multi_task else 'multi'
    replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, 'seed%d' % seed)
    task_idxs = None; store_terminal = None; add_count = None

    if cfg.method.name == 'ARM':
        raise NotImplementedError("ARM is not supported yet")

    elif cfg.method.name == 'BC_LANG':

        if cfg.replay.load_replay_from_disk:
            with open(f"{replay_path}/task_idxs.json", 'r') as f:
                task_idxs = json.load(f)
            assert set(cfg.rlbench.tasks) <= set(list(task_idxs.keys())), \
                "all given task data doesn't exist"
            task_idxs = {task_key: task_idxs[task_key] for task_key in cfg.rlbench.tasks}
            print("replay size with actual data", sum([len(v) for v in task_idxs.values()]), task_idxs.keys())
            store_terminal_ = np.load(f"{replay_path}/store_terminal.npy", allow_pickle=True)
            store_terminal_ = store_terminal_.item()
            store_terminal = store_terminal_
            add_count = len(os.listdir(replay_path))

        assert cfg.ddp.num_devices == 1, "BC_LANG only supports single GPU training"
        replay_buffer = bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.rlbench.camera_resolution,
            task_idxs=task_idxs,
            store_terminal=store_terminal,
            add_count=add_count)

        if not cfg.replay.load_replay_from_disk:
            bc_lang.launch_utils.fill_multi_task_replay(
                cfg, obs_config, rank,
                replay_buffer, tasks, cfg.rlbench.demos,
                cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
                cams)

        agent = bc_lang.launch_utils.create_agent(
            cams[0], cfg.method.activation, cfg.method.lr,
            cfg.method.weight_decay, cfg.rlbench.camera_resolution,
            cfg.method.grad_clip)

    elif cfg.method.name == 'VIT_BC_LANG':
        assert cfg.ddp.num_devices == 1, "VIT_BC_LANG only supports single GPU training"
        if cfg.replay.load_replay_from_disk:
            with open(f"{replay_path}/task_idxs.json", 'r') as f:
                task_idxs = json.load(f)
            assert set(cfg.rlbench.tasks) <= set(list(task_idxs.keys())), \
                "all given task data doesn't exist"
            task_idxs = {task_key: task_idxs[task_key] for task_key in cfg.rlbench.tasks}
            print("replay size with actual data", sum([len(v) for v in task_idxs.values()]), task_idxs.keys())
            store_terminal_ = np.load(f"{replay_path}/store_terminal.npy", allow_pickle=True)
            store_terminal_ = store_terminal_.item()
            store_terminal = store_terminal_
            add_count = len(os.listdir(replay_path))

        replay_buffer = vit_bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.rlbench.camera_resolution,
            task_idxs=task_idxs,
            store_terminal=store_terminal,
            add_count=add_count)

        if not cfg.replay.load_replay_from_disk:
            vit_bc_lang.launch_utils.fill_multi_task_replay(
                cfg, obs_config, rank,
                replay_buffer, tasks, cfg.rlbench.demos,
                cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
                cams)

            with open(f"{replay_path}/task_idxs.json", 'w') as f:
                json.dump(replay_buffer._task_idxs.copy(), f)
            np.save(f"{replay_path}/store_terminal.npy", replay_buffer._store.copy())

            print("replay size with actual+gnerated data", replay_buffer._add_count.value)

        agent = vit_bc_lang.launch_utils.create_agent(
            cams, cfg.method.activation, cfg.method.lr,
            cfg.method.weight_decay, cfg.rlbench.camera_resolution,
            cfg.method.grad_clip)

    elif cfg.method.name == 'C2FARM_LINGUNET_BC':
        replay_buffer = c2farm_lingunet_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution)

        c2farm_lingunet_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = c2farm_lingunet_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'PERACT_BC':

        if cfg.replay.load_replay_from_disk:
            with open(f"{replay_path}/task_idxs.json", 'r') as f:
                task_idxs = json.load(f)
            # print(set(cfg.rlbench.tasks), set(list(task_idxs.keys())))
            # import ipdb; ipdb.set_trace()
            # assert set(cfg.rlbench.tasks) <= set(list(task_idxs.keys())) , \
            #     "all given task data doesn't exist"
            if not set(cfg.rlbench.tasks) <= set(list(task_idxs.keys())):
                cfg.rlbench.tasks = list(task_idxs.keys())
                print(f"changing task list to available tasks (#{len(list(task_idxs.keys()))})")
            else:
                task_idxs = {task_key: task_idxs[task_key] for task_key in cfg.rlbench.tasks}
            print(replay_path)
            print("replay size with actual data", sum([len(v) for v in task_idxs.values()]))
            store_terminal_ = np.load(f"{replay_path}/store_terminal.npy", allow_pickle=True)
            store_terminal_ = store_terminal_.item()
            #TODO: edit store_terminal based on task idxs len, since it's one array
            ## only works with fixed num of demos when editing
            ### no need to do this since indices correspond to the store_terminal arr indices
            ### same reason for giving total count in the replay dir
            store_terminal = store_terminal_
            # store_terminal['terminal'] = np.full(store_terminal_['terminal'].shape, -1)
            # idx = 0
            # for k, v in task_idxs.items():
            #     store_terminal['terminal'][idx : idx+len(v)] = store_terminal_['terminal'][v[0]: v[-1]+1]
            #     idx += len(v)
            add_count = sum([len(v) for v in task_idxs.values()])

            # add_count = len(os.listdir(replay_path))
        # print('Task ids:', task_idxs)
        # print('add count:', add_count)
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
            task_idxs=task_idxs,
            store_terminal=store_terminal,
            add_count=add_count
            )

        if not cfg.replay.load_replay_from_disk:
            peract_bc.launch_utils.fill_multi_task_replay(
                cfg, obs_config, rank,
                replay_buffer, tasks, cfg.rlbench.demos,
                cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
                cams, cfg.rlbench.scene_bounds,
                cfg.method.voxel_sizes, cfg.method.bounds_offset,
                cfg.method.rotation_resolution, cfg.method.crop_augmentation,
                keypoint_method=cfg.method.keypoint_method)
        # if not cfg.replay.load_replay_from_disk:
        #     peract_bc.launch_utils.fill_replay(
        #         cfg, obs_config, rank,
        #         replay_buffer, tasks[0], cfg.rlbench.demos,
        #         cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
        #         cams, cfg.rlbench.scene_bounds,
        #         cfg.method.voxel_sizes, cfg.method.bounds_offset,
        #         cfg.method.rotation_resolution, cfg.method.crop_augmentation, torch.device('cuda'),
        #         keypoint_method=cfg.method.keypoint_method)


            with open(f"{replay_path}/task_idxs.json", 'w') as f:
                json.dump(replay_buffer._task_idxs.copy(), f)
            np.save(f"{replay_path}/store_terminal.npy", replay_buffer._store.copy())

            print("replay size with actual+gnerated data", replay_buffer._add_count.value)
        
        agent = peract_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'PERACT_RL':
        raise NotImplementedError("PERACT_RL is not supported yet")

    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    # import ipdb; ipdb.set_trace()
    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    # dataloader = wrapped_replay.dataset()
    # print(len(dataloader))
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    logdir = os.path.join(cwd, 'seed%d' % seed)


    ## TODO: load pretrained agent
    # already loads from last existing weight

    ## TODO: generate data and add to 30 points to replay buffer
    ## peract_bc.launch_utils.fill_replay_with_generated_data(agent=agent)
    ## train again from scratch with demo+gen

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size) ##replay_buffer_augmented


    

    # train_runner._agent = copy.deepcopy(train_runner._agent)
    # train_runner._agent.build(training=True, device=train_runner._train_device)
    
    # train_runner.generate_data()
    # wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers) ##
    # train_runner._wrapped_buffer = wrapped_replay

    # train_runner._iterations = 100000
    train_runner._weightsdir = weightsdir
    # print("replay size with actual+gnerated data", replay_buffer._add_count.value)
    # print("replay size with actual+gnerated data", replay_buffer._add_count.value) ##

    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()