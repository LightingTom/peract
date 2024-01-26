# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

import logging
from typing import List
import random

import numpy as np
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import demo_loading_utils, utils
from helpers.preprocess_agent import PreprocessAgent
from helpers.clip.core.clip import tokenize
from agents.peract_bc.perceiver_lang_io import PerceiverVoxelLangEncoder
from agents.peract_bc.qattention_peract_bc_agent import QAttentionPerActBCAgent
from agents.peract_bc.qattention_stack_agent import QAttentionStackAgent

import torch
import torch.nn as nn
import multiprocessing as mp
from torch.multiprocessing import Process, Value, Manager
from helpers.clip.core.clip import build_model, load_clip, tokenize
from omegaconf import DictConfig

REWARD_SCALE = 100.0
LOW_DIM_SIZE = 4


def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  voxel_sizes,
                  image_size=[128, 128],
                  replay_size=3e5,
                  task_idxs=None,
                  store_terminal=None,
                  add_count=None):
    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512
    num_demo = 10

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

        observation_elements.append(
            ObservationElement('tar_%s_rgb' % cname, (num_demo - 1, 3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('tar_%s_point_cloud' % cname, (num_demo - 1, 3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('tar_%s_camera_extrinsics' % cname, (num_demo - 1, 4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('tar_%s_camera_intrinsics' % cname, (num_demo - 1, 3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (num_demo - 1, trans_indicies_size,),
                      np.int32),
        ReplayElement('rot_grip_action_indicies', (num_demo - 1, rot_and_grip_indicies_size,),
                      np.int32),
        ReplayElement('ignore_collisions', (num_demo - 1, ignore_collisions_size,),
                      np.float32),
        ReplayElement('noises', (num_demo - 1, 64, 64),
                      np.int32),
        ReplayElement('gripper_pose', (num_demo - 1, gripper_pose_size,),
                      np.float32),
        ReplayElement('lang_goal_emb', (lang_feat_dim,),
                      np.float32),
        ReplayElement('lang_token_embs', (max_token_seq_len, lang_emb_dim,),
                      np.float32),  # extracted from CLIP's language encoder
        ReplayElement('task', (),
                      str),
        ReplayElement('lang_goal', (1,),
                      object),  # language goal string for debugging and visualization
        # ReplayElement('voxel_reconstructed', (10, voxel_sizes[0], voxel_sizes[0], voxel_sizes[0]),
        #               np.float32),
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
        ReplayElement('test_task', (), np.bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements,
    )
    return replay_buffer


def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,
        rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes):  # only single voxelization-level is used in PerAct
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                     attention_coordinate + bounds_offset[depth - 1]])
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates


def _add_keypoints_to_replay(
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        inital_obs: Observation,
        tar_obs_list,
        demo: Demo,
        tar_demo_list,
        episode_keypoints: List[int],
        tar_episode_keypoints_list,
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        description: str = '',
        clip_model=None,
        device='cpu'):
    prev_action = [None for _ in range(len(tar_demo_list))]
    prev_action_ = None
    obs = inital_obs
    noises = np.random.normal(0, 1, size=(10-1, 64, 64))
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        tar_kp_list = []
        tar_tp1_list = []
        tar_tm1_list = []
        for idx in range(len(tar_demo_list)):
            if k >= len(tar_episode_keypoints_list[idx]) - 1:
                tar_kp = keypoint
            else:
                tar_kp = tar_episode_keypoints_list[idx][k]
            tar_kp_list.append(tar_kp)
            if tar_kp_list[-1] >= len(tar_demo_list[idx]):
                tar_tp1 = tar_demo_list[idx][tar_episode_keypoints_list[idx][-1]]
            else:
                tar_tp1 = tar_demo_list[idx][tar_kp_list[-1]]
            tar_tp1_list.append(tar_tp1)
            if tar_kp_list[-1] >= len(tar_demo_list[idx]):
                tar_tm1 = tar_demo_list[idx][tar_episode_keypoints_list[idx][-1]]
            else:
                tar_tm1 = tar_demo_list[idx][max(0, tar_kp_list[-1] - 1)]
            tar_tm1_list.append(tar_tm1)
        obs_tm1 = demo[max(0, keypoint - 1)]
        # ori
        _, _, _, action_, _ = _get_action(obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes, bounds_offset,
                                          rotation_resolution, crop_augmentation)
        # ver 1221
        # trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
        #     tar_tp1, tar_tm1, rlbench_scene_bounds, voxel_sizes, bounds_offset,
        #     rotation_resolution, crop_augmentation)
        trans_list = []
        rot_grip_list = []
        ignore_collision_list = []
        action_list = []
        attention_coor_list = []
        for idx in range(len(tar_demo_list)):
            trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
                tar_tp1_list[idx], tar_tm1_list[idx], rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation)
            trans_list.append(trans_indicies)
            rot_grip_list.append(rot_grip_indicies)
            ignore_collision_list.append(ignore_collisions)
            action_list.append(action)
            attention_coor_list.append(attention_coordinates)


        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = utils.extract_obs(obs, t=k, prev_action=prev_action_,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length, name='ori')
        tar_list = []
        for idx in range(len(tar_obs_list)):
            tar_dict = utils.extract_obs(tar_obs_list[idx], t=k, prev_action=prev_action[idx],
                                         cameras=cameras, episode_length=cfg.rlbench.episode_length, name='tar')
            tar = {}
            for key in tar_dict.keys():
                if key not in ['low_dim_state']:
                    tar['tar_' + key] = tar_dict[key]
            tar_list.append(tar)
        tar_combine = {}
        for tar_key in tar_list[0].keys():
            items = []
            for idx in range(len(tar_list)):
                items.append(tar_list[idx][tar_key])
            tar_combine[tar_key] = np.stack(items, axis=0)
        # tar_combine['ignore_collisions'] = np.stack(ignore_collision_list, axis=0).reshape((-1, 1))
        tar_combine['ignore_collisions'] = tar_combine.pop('tar_ignore_collisions')
        tokens = tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        obs_dict['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
        obs_dict['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()
        if np.isnan(obs_dict['lang_goal_emb'])[0]:
            print('clip nan', description)
            continue

        prev_action_ = np.copy(action_)
        for idx in range(len(prev_action)):
            prev_action[idx] = np.copy(action_list[idx])

        others = {'demo': True,
                  'test_task': True if 'test' in task else False
                  }

        gripper_pose_list = []
        for idx in range(len(tar_tp1_list)):
            gripper_pose_list.append(tar_tp1_list[idx].gripper_pose)
        final_obs = {
            'trans_action_indicies': np.stack(trans_list, axis=0),
            'rot_grip_action_indicies': np.stack(rot_grip_list, axis=0),
            'gripper_pose': np.stack(gripper_pose_list, axis=0),
            'noises': noises,
            'task': task,
            'lang_goal': np.array([description], dtype=object),
        }
        final_obs.update(tar_combine)

        others.update(obs_dict)
        others.update(final_obs)
        # import ipdb; ipdb.set_trace()
        timeout = False
        replay.add(action_, reward, terminal, timeout, **others)
        obs = obs_tp1
        tar_obs_list = tar_tp1_list

    # final step
    obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1, prev_action=prev_action_,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)

    tar_dict_tp1_list = []
    for idx in range(len(tar_tp1_list)):
        tar_dict_tp1 = utils.extract_obs(tar_tp1_list[idx], t=k + 1, prev_action=prev_action,
                                         cameras=cameras, episode_length=cfg.rlbench.episode_length)
        tar_tp1 = {}
        for key in obs_dict_tp1.keys():
            if key not in ['low_dim_state']:
                tar_tp1['tar_' + key] = tar_dict_tp1[key]
        tar_dict_tp1_list.append(tar_tp1)
    tar_tp1_combine = {}
    for tar_key in tar_dict_tp1_list[0].keys():
        items = []
        for idx in range(len(tar_dict_tp1_list)):
            items.append(tar_dict_tp1_list[idx][tar_key])
        tar_tp1_combine[tar_key] = np.stack(items, axis=0)
    tar_tp1_combine['ignore_collisions'] = tar_tp1_combine.pop('tar_ignore_collisions')
    obs_dict_tp1['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop('wrist_world_to_cam', None)
    final_obs.update(tar_tp1_combine)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)


def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                clip_model=None,
                device='cpu',
                keypoint_method='heuristic'):
    logging.getLogger().setLevel(cfg.framework.logging_level)

    if clip_model is None:
        model, _ = load_clip('RN50', jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug('Filling %s replay ...' % task)
    for d_idx in range(num_demos):
        # load demo from disk
        demo = rlbench_utils.get_stored_demos(
            amount=1, image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1, task_name=task,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=d_idx,
            name='ori_obs')[0]
        # load the target image
        # randomly get a new directory
        tar_demo_list = []
        for i in range(num_demos):
            if i == d_idx:
                continue
            target = rlbench_utils.get_stored_demos(
                    amount=1, image_paths=False,
                    dataset_root=cfg.rlbench.demo_path,
                    variation_number=-1, task_name=task,
                    obs_config=obs_config,
                    random_selection=False,
                    from_episode_number=i,
                    name='tar_obs')[0]
            tar_demo_list.append(target)

        descs = demo._observations[0].misc['descriptions']

        # extract keypoints (a.k.a keyframes)
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo, method=keypoint_method)
        tar_episode_keypoints_list = []
        for i in range(len(tar_demo_list)):
            tar_episode_keypoints_list.append(
                demo_loading_utils.keypoint_discovery(tar_demo_list[i], method=keypoint_method))

        if rank == 0:
            logging.info(f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints - {task}")

        # tar_demo = demo
        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue
            # import ipdb; ipdb.set_trace()
            obs = demo[i]
            # get the target demo according to the ratio
            tar_list = []
            tar_zero = False
            for t_idx in range(len(tar_demo_list)):
                frag = len(tar_demo_list[t_idx]) / len(demo)
                idx = int(i * frag)
                tar_list.append(tar_demo_list[t_idx][idx])
                for kp_idx in range(len(tar_episode_keypoints_list)):
                    while len(tar_episode_keypoints_list[kp_idx]) > 0 and idx >= tar_episode_keypoints_list[kp_idx][0]:
                        tar_episode_keypoints_list[kp_idx] = tar_episode_keypoints_list[kp_idx][1:]
                    if len(tar_episode_keypoints_list[kp_idx]) == 0:
                        tar_zero = True
            desc = random.choice(descs)
            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            if tar_zero:
                break
            _add_keypoints_to_replay(
                cfg, task, replay, obs, tar_list, demo, tar_demo_list,
                episode_keypoints, tar_episode_keypoints_list,
                cameras, rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, description=desc,
                clip_model=clip_model, device=device)
    logging.info('Replay %s filled with demos.' % task)


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig,
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str],
                           rlbench_scene_bounds: List[float],
                           voxel_sizes: List[int],
                           bounds_offset: List[float],
                           rotation_resolution: int,
                           crop_augmentation: bool,
                           clip_model=None,
                           keypoint_method='heuristic'):
    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)

    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
                                        if torch.cuda.is_available() else 'cpu')
            p = Process(target=fill_replay, args=(cfg,
                                                  obs_config,
                                                  rank,
                                                  replay,
                                                  task,
                                                  num_demos,
                                                  demo_augmentation,
                                                  demo_augmentation_every_n,
                                                  cameras,
                                                  rlbench_scene_bounds,
                                                  voxel_sizes,
                                                  bounds_offset,
                                                  rotation_resolution,
                                                  crop_augmentation,
                                                  clip_model,
                                                  model_device,
                                                  keypoint_method))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # for split in split_n:
    #     for e_idx, task_idx in enumerate(split):
    #         task = tasks[int(task_idx)]
    #         model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
    #                                     if torch.cuda.is_available() else 'cpu')
    #         fill_replay(cfg,
    #                     obs_config,
    #                     rank,
    #                     replay,
    #                     task,
    #                     num_demos,
    #                     demo_augmentation,
    #                     demo_augmentation_every_n,
    #                     cameras,
    #                     rlbench_scene_bounds,
    #                     voxel_sizes,
    #                     bounds_offset,
    #                     rotation_resolution,
    #                     crop_augmentation,
    #                     clip_model,
    #                     model_device,
    #                     keypoint_method)


def create_agent(cfg: DictConfig):
    LATENT_SIZE = 64
    depth_0bounds = cfg.rlbench.scene_bounds
    cam_resolution = cfg.rlbench.camera_resolution

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        last = depth == len(cfg.method.voxel_sizes) - 1
        perceiver_encoder = PerceiverVoxelLangEncoder(
            depth=cfg.method.transformer_depth,
            iterations=cfg.method.transformer_iterations,
            voxel_size=vox_size,
            initial_dim=3 + 3 + 1 + 3,
            low_dim_size=4,
            # it should be 0 for gen data since its noisy for grip, finger positions, and timestep is also not known
            layer=depth,
            num_rotation_classes=num_rotation_classes if last else 0,
            num_grip_classes=2 if last else 0,
            num_collision_classes=2 if last else 0,
            input_axis=3,
            num_latents=cfg.method.num_latents,
            latent_dim=cfg.method.latent_dim,
            cross_heads=cfg.method.cross_heads,
            latent_heads=cfg.method.latent_heads,
            cross_dim_head=cfg.method.cross_dim_head,
            latent_dim_head=cfg.method.latent_dim_head,
            weight_tie_layers=False,
            activation=cfg.method.activation,
            pos_encoding_with_lang=cfg.method.pos_encoding_with_lang,
            input_dropout=cfg.method.input_dropout,
            attn_dropout=cfg.method.attn_dropout,
            decoder_dropout=cfg.method.decoder_dropout,
            lang_fusion_type=cfg.method.lang_fusion_type,
            voxel_patch_size=cfg.method.voxel_patch_size,
            voxel_patch_stride=cfg.method.voxel_patch_stride,
            no_skip_connection=cfg.method.no_skip_connection,
            no_perceiver=cfg.method.no_perceiver,
            no_language=cfg.method.no_language,
            final_dim=cfg.method.final_dim,
            reconstruction3D_pretraining=cfg.method.reconstruction3D_pretraining,
            decode_occupnacy=cfg.method.decode_occupnacy,
            decode_rgb=cfg.method.decode_rgb,
            factor=cfg.method.factor
        )

        qattention_agent = QAttentionPerActBCAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            perceiver_encoder=perceiver_encoder,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            lr=cfg.method.lr,
            training_iterations=cfg.framework.training_iterations,
            lr_scheduler=cfg.method.lr_scheduler,
            num_warmup_steps=cfg.method.num_warmup_steps,
            trans_loss_weight=cfg.method.trans_loss_weight,
            rot_loss_weight=cfg.method.rot_loss_weight,
            grip_loss_weight=cfg.method.grip_loss_weight,
            collision_loss_weight=cfg.method.collision_loss_weight,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            transform_augmentation=cfg.method.transform_augmentation.apply_se3,
            transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
            transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
            transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
            optimizer_type=cfg.method.optimizer,
            num_devices=cfg.ddp.num_devices,
            reconstruction3D_pretraining=cfg.method.reconstruction3D_pretraining,
            masked_decoding=cfg.method.masked_decoding,
            masking_ratio=cfg.method.masking_ratio,
            masking_type=cfg.method.masking_type,
            input_masking_ratio=cfg.method.input_masking_ratio,
            train_with_seen_objects=cfg.method.train_with_seen_objects
        )
        qattention_agents.append(qattention_agent)

    # import ipdb; ipdb.set_trace()
    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )
    preprocess_agent = PreprocessAgent(
        pose_agent=rotation_agent
    )
    return preprocess_agent


def fill_replay_with_generated_data():
    # TODO: call model with a parameter that adds noise after cross attention
    # also needs input here to add noise to
    raise NotImplementedError("aug replay not implemented")
