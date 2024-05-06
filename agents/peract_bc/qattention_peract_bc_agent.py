import copy
import logging
import os
from typing import List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pytorch3d import transforms as torch3d_tf
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from helpers import utils
from helpers.utils import visualise_voxel, stack_on_channel
from voxel.voxel_grid import VoxelGrid
from voxel.augmentation import apply_se3_augmentation
from einops import rearrange
from helpers.clip.core.clip import build_model, load_clip
from torchvision.utils import save_image

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import transformers
from helpers.optim.lamb import Lamb
from patchify import patchify, unpatchify

from torch.nn.parallel import DistributedDataParallel as DDP
import pickle

NAME = 'QAttentionAgent'


# torch.autograd.detect_anomaly()

class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxelizer: VoxelGrid,
                 bounds_offset: float,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)
        self.training = training

        # distributed training
        if training:
            self._qnet = DDP(self._qnet, device_ids=[device], find_unused_parameters=True)

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self, rgb_pcd, tar_rgb_pcd, proprio, pcd, tar_pcd, lang_goal_emb, lang_token_embs,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None,
                generate=0, noisy_voxel_grid=None, masked_decoding=False,
                masking_ratio=0.8, input_masking_ratio=0.5, masking_type='patch', noise=None,
                vae=True, enc=True):
        # rgb_pcd will be list of list (list of [rgb, pcd])

        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        tar_b = tar_rgb_pcd[0][0].shape[0]
        tar_pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(tar_b, -1, 3) for p in tar_pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)

        tar_rgb = [rp[0] for rp in tar_rgb_pcd]
        tar_feat_size = tar_rgb[0].shape[1]
        tar_flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(tar_b, -1, tar_feat_size) for p in tar_rgb], 1)

        # construct voxel grid
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)
        tar_voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            tar_pcd_flat, coord_features=tar_flat_imag_features, coord_bounds=bounds)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()
        tar_voxel_grid = tar_voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # if not generate:
        #     import ipdb; ipdb.set_trace()
        ## use genrate voxel grid if this is an augmented noisy point
        voxel_grid = noisy_voxel_grid.detach() if noisy_voxel_grid is not None else voxel_grid
        # print(f"in qnet, generate={generate}", noisy_voxel_grid, voxel_grid.shape)

        device = voxel_grid.device
        # masking based on the occupancy channel
        # print('HERE masked_decoding', masked_decoding)
        if not masked_decoding:
            voxel_grid_masked = None
        elif masked_decoding and self.training and random.randint(1, 100) <= int(
                input_masking_ratio * 100):  # 50% input masking
            if masking_type == 'voxel':
                mask = (torch.FloatTensor(voxel_grid.shape[0], 1, voxel_grid.shape[-1], voxel_grid.shape[-1],
                                          voxel_grid.shape[-1]).uniform_() > masking_ratio).to(device)
                mask = mask.repeat(1, voxel_grid.shape[1], 1, 1, 1)
                voxel_grid_masked = voxel_grid * mask.int().float()
            if masking_type == 'patch':
                # TODO: currently same patch mask for all batch items
                mask = (torch.ones(voxel_grid.shape[-1], voxel_grid.shape[-1], voxel_grid.shape[-1])).to(device)
                # import ipdb; ipdb.set_trace()
                patches = torch.from_numpy(patchify(mask.cpu().numpy(), (5, 5, 5), step=5)).to(device)
                patches = patches.flatten(start_dim=0, end_dim=2)
                indices = torch.randint(20 ** 3, (int(20 ** 3 * masking_ratio),)).to(device)
                patches[indices] *= 0.
                patches = patches.unflatten(0, (20, 20, 20))
                mask = torch.from_numpy(unpatchify(patches.cpu().numpy(), mask.shape)).to(device)
                mask = mask.unsqueeze(0).unsqueeze(0).repeat(voxel_grid.shape[0], voxel_grid.shape[1], 1, 1, 1)
                voxel_grid_masked = voxel_grid * mask.int().float()
            else:
                raise Exception('Unknown masking type')
        else:
            voxel_grid_masked = voxel_grid

        # forward pass
        q_trans, \
        q_rot_and_grip, \
        q_ignore_collisions, voxel_reconstructed,\
        vae_mean, vae_var,\
        x_mean, x_var = self._qnet(
            voxel_grid_masked if masked_decoding else voxel_grid,
            proprio,
            lang_goal_emb,
            lang_token_embs,
            prev_layer_voxel_grid,
            bounds,
            prev_bounds,
            generate=generate,
            noise=noise,
            vae=vae,
            enc=enc)

        # if generate:
        #     voxel_grid = voxel_reconstructed

        return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid, \
               voxel_grid_masked, voxel_reconstructed, tar_voxel_grid, \
               vae_mean, vae_var, x_mean, x_var


class QAttentionPerActBCAgent(Agent):

    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 perceiver_encoder: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 lr: float = 0.0001,
                 lr_scheduler: bool = False,
                 training_iterations: int = 100000,
                 num_warmup_steps: int = 20000,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 transform_augmentation: bool = True,
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'adam',
                 num_devices: int = 1,
                 reconstruction3D_pretraining: bool = False,
                 masked_decoding=True,
                 masking_ratio=0.8,
                 masking_type='patch',
                 input_masking_ratio=0.5,
                 train_with_seen_objects=False,
                 vae=False
                 ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self.reconstruction3D_pretraining = reconstruction3D_pretraining

        self.masked_decoding = masked_decoding
        self.masking_ratio = masking_ratio
        self.masking_type = masking_type
        self.input_masking_ratio = input_masking_ratio

        self.train_with_seen_objects = train_with_seen_objects

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._l1_loss = nn.L1Loss(reduction='mean')
        self._l2_loss = nn.MSELoss(reduction='mean')
        self._bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self._name = NAME + '_layer' + str(self._layer)
        self._vae = vae

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if device is None:
            device = torch.device('cpu')

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        self._q = QFunction(self._perceiver_encoder,
                            self._voxelizer,
                            self._bounds_offset,
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        grid_for_crop = torch.arange(0,
                                     self._image_crop_size,
                                     device=device).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._training_iterations // 10000,
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                            1,
                                                            self._voxel_size,
                                                            self._voxel_size,
                                                            self._voxel_size),
                                                           dtype=int,
                                                           device=device)
            self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                           dtype=int,
                                                           device=device)
            self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                           dtype=int,
                                                           device=device)
            self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                           dtype=int,
                                                           device=device)
            self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                           2),
                                                          dtype=int,
                                                          device=device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                        2),
                                                                       dtype=int,
                                                                       device=device)

            # print total params
            logging.info('# Q Params: %d' % sum(
                p.numel() for name, p in self._q.named_parameters() \
                if p.requires_grad and 'clip' not in name))
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._voxelizer.to(device)
            self._q.to(device)

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        # observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0,
            h - self._image_crop_size)
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample, vae):
        obs = []
        pcds = []

        tar_obs = []
        tar_pcds = []
        self._crop_summary = []
        for n in self._camera_names:
            rgb = replay_sample['%s_rgb' % n]
            pcd = replay_sample['%s_point_cloud' % n]
            if vae:
                tar_rgb = rgb.clone().detach()
                tar_pcd = pcd.clone().detach()
                # print('equal:', torch.equal(rgb, tar_rgb))
            else:
                tar_rgb = replay_sample['tar_%s_rgb' % n]
                tar_pcd = replay_sample['tar_%s_point_cloud' % n]
            # print('rgb:', torch.equal(rgb, tar_rgb))
            # print('pcd:', torch.equal(pcd, tar_pcd))
            # print('ori:', pcd.shape)
            # print('tar:', tar_pcd.shape)
            obs.append([rgb, pcd])
            tar_obs.append([tar_rgb, tar_pcd])
            pcds.append(pcd)
            tar_pcds.append(tar_pcd)
        return obs, pcds, tar_obs, tar_pcds

    def _act_preprocess_inputs(self, observation):
        # observation keys:
        # 'left_shoulder_rgb', 'left_shoulder_point_cloud', 'right_shoulder_rgb', 'right_shoulder_point_cloud',
        # 'wrist_rgb', 'wrist_point_cloud', 'front_rgb', 'front_point_cloud', 'ignore_collisions', 'low_dim_state',
        # 'left_shoulder_camera_extrinsics', 'left_shoulder_camera_intrinsics', 'right_shoulder_camera_extrinsics',
        # 'right_shoulder_camera_intrinsics', 'front_camera_extrinsics', 'front_camera_intrinsics', 'wrist_camera_extrinsics',
        # 'wrist_camera_intrinsics', 'lang_goal_tokens'
        # print(observation['front_rgb'].shape)
        # 1,1,3,128,128
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = observation['%s_rgb' % n].squeeze(0)
            pcd = observation['%s_point_cloud' % n].squeeze(0)

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(torch.split(
            rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
            dim=1), dim=1)  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
             q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
             q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
             q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0 * self._num_rotation_classes: 1 * self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1 * self._num_rotation_classes: 2 * self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2 * self._num_rotation_classes: 3 * self._num_rotation_classes]
        q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def get_tar_voxel_grid(self, rgb_pcd, pcd):
        bounds = self._coordinate_bounds.to(self._device)
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)

        # construct voxel grid
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)
        return voxel_grid.permute(0, 4, 1, 2, 3).detach()

    def visualize_latent_space(self, mean, var, step, enc):
        # print('enc:', enc)
        np_mean = np.squeeze(mean.clone().detach().cpu().numpy(), axis=-1)[0]
        np_var = np.squeeze(var.clone().detach().cpu().numpy(), axis=-1)[0]
        if enc:
            np_var = np.exp(np_var)
        # print('mean:', np_mean.shape)
        # print('var:', np_var.shape)
        covariance = np_var * np.identity(64)
        samples = np.random.multivariate_normal(np_mean, covariance, 1000)
        pca = PCA(n_components=3)
        reduced_samples = pca.fit_transform(samples)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(reduced_samples[:, 0], reduced_samples[:, 1], reduced_samples[:, 2],
                   c='blue', marker='o', alpha=0.5)

        ax.set_title('Latent space in 3D')
        plt.savefig('%d.png' % step)

    def update(self, step: int, replay_sample: dict) -> dict:
        # print('step:', step)

        num_demos = 10
        choice = random.randint(0, num_demos - 2)
        # this four is for target (output) img
        trans = replay_sample['trans_action_indicies'].permute(1, 0, 2)
        rot = replay_sample['rot_grip_action_indicies'].permute(1, 0, 2)
        gripper = replay_sample['gripper_pose'].permute(1, 0, 2)
        ig_col = replay_sample['ignore_collisions'].permute(1, 0, 2)

        action_trans = trans[choice][:, self._layer * 3:self._layer * 3 + 3].int()
        action_rot_grip = rot[choice].int()
        action_gripper_pose = gripper[choice]
        action_ignore_collisions = ig_col[choice].int()
        lang_goal_emb = replay_sample['lang_goal_emb'].float()
        lang_token_embs = replay_sample['lang_token_embs'].float()
        prev_layer_voxel_grid = replay_sample.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = replay_sample.get('prev_layer_bounds', None)
        noise = replay_sample['noises'].permute(1, 0, 2, 3)[choice]
        # print('noise:', noise.shape)
        device = self._device
        # print(os.getcwd())
        # if not os.path.exists('errors') and len(error_terms) != 0:
        #     with open('errors', 'wb') as f:
        #         pickle.dump(error_terms, f)
        #         print('finish writing')
        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)]
            bounds = torch.cat([cp - self._bounds_offset, cp + self._bounds_offset], dim=1)

        proprio = None
        if self._include_low_dim_state:
            proprio = replay_sample['low_dim_state']

        for cname in self._camera_names:
            replay_sample.pop('tar_%s_rgb_tp1' % cname)
            replay_sample.pop('tar_%s_point_cloud_tp1' % cname)

        for cname in self._camera_names:
            replay_sample['tar_%s_rgb' % cname] = replay_sample['tar_%s_rgb' % cname].permute(1, 0, 2, 3, 4)[choice]
            replay_sample['tar_%s_point_cloud' % cname] = replay_sample['tar_%s_point_cloud' % cname].permute(1, 0, 2, 3, 4)[choice]

        # if we use VAE encoder
        enc = False
        obs, pcd, tar_obs, tar_pcd = self._preprocess_inputs(replay_sample, enc)

        # batch size
        bs = pcd[0].shape[0]

        # import ipdb; ipdb.set_trace()
        # SE(3) augmentation of point clouds and actions
        # don't add augmentation
        self._transform_augmentation = False
        if self._transform_augmentation:
            action_trans, \
            action_rot_grip, \
            pcd = apply_se3_augmentation(pcd,
                                         action_gripper_pose,
                                         action_trans,
                                         action_rot_grip,
                                         bounds,
                                         self._layer,
                                         self._transform_augmentation_xyz,
                                         self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size,
                                         self._rotation_resolution,
                                         self._device)

        #
        # generated_voxel_grid_data_point = True if replay_sample['voxel_reconstructed'].sum().item() != 0 else False

        # voxel_grid = replay_sample['voxel_reconstructed'] if generated_voxel_grid_data_point else None
        voxel_grid = None

        # forward pass
        q_trans, q_rot_grip, \
        q_collision, \
        voxel_grid, \
        voxel_grid_masked, \
        voxel_reconstructed, \
        tar_voxel_grid,\
        vae_mean, vae_var,\
        x_mean, x_var = self._q(obs,
                                 tar_obs,
                                 proprio,
                                 pcd,
                                 tar_pcd,
                                 lang_goal_emb,
                                 lang_token_embs,
                                 bounds,
                                 prev_layer_bounds,
                                 prev_layer_voxel_grid,
                                 noisy_voxel_grid=voxel_grid,
                                 masked_decoding=self.masked_decoding,
                                 masking_ratio=self.masking_ratio,
                                 masking_type=self.masking_type,
                                 input_masking_ratio=self.input_masking_ratio,
                                 noise=noise,
                                 vae=self._vae,
                                 enc=enc)

        # print('use vae:', self._vae)
        if self._vae and step % 1000 == 0:
            print('generating latent space')
            self.visualize_latent_space(vae_mean, vae_var, step, enc)

        # print(torch.equal(voxel_grid, tar_voxel_grid))
        if not self.reconstruction3D_pretraining:
            # argmax to choose best action
            coords, \
            rot_and_grip_indicies, \
            ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                      q_rot_grip,
                                                                      q_collision)

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss, \
        q_recons_loss = 0., 0., 0., 0., 0.

        # import ipdb; ipdb.set_trace()
        # voxel_grid = self.get_tar_voxel_grid(obs, pcd)
        ## loss computed with unmasked grid
        # only need to do this loss when GT voxel grid is there, which means no recons grid
        # add reconstruction loss if reconstructing GT input
        if voxel_reconstructed != None:
            # pixelwise loss 500:10 ratio
            loss_mask = tar_voxel_grid[:, -1].bool()
            q_recons_loss += 500 * self._bce_loss(voxel_reconstructed[:, -1][loss_mask],
                                                  tar_voxel_grid[:, -1][loss_mask])
            q_recons_loss += 10 * self._bce_loss(voxel_reconstructed[:, -1][~loss_mask],
                                                 tar_voxel_grid[:, -1][~loss_mask])

            if voxel_reconstructed.shape[1] > 1:  ## rgb channels are predicted
                loss_mask = loss_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
                q_recons_loss += 100 * self._l2_loss(voxel_reconstructed[:, 0:3][loss_mask],
                                                     tar_voxel_grid[:, 3:6][loss_mask])
                q_recons_loss += 10 * self._l2_loss(voxel_reconstructed[:, 0:3][~loss_mask],
                                                    tar_voxel_grid[:, 3:6][~loss_mask])

        vae_kl_divergence = 0.5 * torch.sum(-1 - vae_var + torch.exp(vae_var) + vae_mean**2)
        # print('kl-divergence', vae_kl_divergence.isnan())
        if self.reconstruction3D_pretraining:
            ## skip computing action losses
            total_loss = q_recons_loss + vae_kl_divergence

        else:
            # TODO: add an arg for this, if it works
            # import ipdb; ipdb.set_trace()
            with_rot_and_grip = rot_and_grip_indicies is not None

            # translation one-hot
            action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
            for b in range(bs):
                gt_coord = action_trans[b, :].int()
                action_trans_one_hot[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

            # translation loss
            q_trans_flat = q_trans.view(bs, -1)
            action_trans_one_hot_flat = action_trans_one_hot.view(bs, -1)
            q_trans_loss = self._celoss(q_trans_flat, action_trans_one_hot_flat)
            # q_trans_loss = q_trans_loss.mean()

            if with_rot_and_grip:
                # rotation, gripper, and collision one-hots
                action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
                action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
                action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
                action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
                action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

                for b in range(bs):
                    gt_rot_grip = action_rot_grip[b, :].int()
                    action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                    action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                    action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                    action_grip_one_hot[b, gt_rot_grip[3]] = 1

                    gt_ignore_collisions = action_ignore_collisions[b, :].int()
                    action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

                # flatten predictions
                q_rot_x_flat = q_rot_grip[:, 0 * self._num_rotation_classes:1 * self._num_rotation_classes]
                q_rot_y_flat = q_rot_grip[:, 1 * self._num_rotation_classes:2 * self._num_rotation_classes]
                q_rot_z_flat = q_rot_grip[:, 2 * self._num_rotation_classes:3 * self._num_rotation_classes]
                q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes:]
                q_ignore_collisions_flat = q_collision

                # rotation loss
                q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
                q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
                q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)
                # q_rot_loss = q_rot_loss.mean()

                # gripper loss
                q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)
                # q_grip_loss = q_grip_loss.mean()

                # collision loss
                q_collision_loss += self._celoss(q_ignore_collisions_flat, action_ignore_collisions_one_hot)
                # q_collision_loss = q_collision_loss.mean()

            if self.train_with_seen_objects:
                mask = (1 - replay_sample['test_task'].detach())
                combined_losses = (q_trans_loss * self._trans_loss_weight * mask) + \
                                  (q_rot_loss * self._rot_loss_weight * mask) + \
                                  (q_grip_loss * self._grip_loss_weight * mask) + \
                                  (q_collision_loss * self._collision_loss_weight * mask) + \
                                  (q_recons_loss) + (vae_kl_divergence)
            else:
                combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                                  (q_rot_loss * self._rot_loss_weight) + \
                                  (q_grip_loss * self._grip_loss_weight) + \
                                  (q_collision_loss * self._collision_loss_weight) + \
                                  (q_recons_loss) + (vae_kl_divergence)

            total_loss = combined_losses.mean()

        if total_loss.isnan().item():
            import ipdb;
            ipdb.set_trace()

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        # import ipdb; ipdb.set_trace()
        ## using unmasked grid to compute error in reconstruction
        if self.reconstruction3D_pretraining:
            self._summaries = {
                'losses/total_loss': total_loss,
                'losses/recons_loss': q_recons_loss,
                'losses/vae_kl_divergence': vae_kl_divergence,
                'info/x_mean': 0,
                'info/x_var': 0,
                'info/ocupied_voxel_pred': (
                        voxel_reconstructed[0, -1] >= 0.5).sum() if voxel_reconstructed != None else 0.,
                'info/ocupied_voxel': tar_voxel_grid[0, -1].sum(),
                'info/scene_diff_occup_wThreshold': voxel_grid[0, -1].sum() - (
                        voxel_reconstructed[0, -1] >= 0.5).sum() if voxel_reconstructed != None else 0.,
                'info/scene_diff_occup': tar_voxel_grid[0, -1].sum() - (
                    voxel_reconstructed[0, -1]).sum() if voxel_reconstructed != None else 0.,
                'info/scene_diff_RGB': tar_voxel_grid[0, 3:6].abs().sum() - voxel_reconstructed[0,
                                                                            0:3].abs().sum() if voxel_reconstructed != None else 0.,
            }
        else:
            self._summaries = {
                'losses/total_loss': total_loss,
                'losses/trans_loss': q_trans_loss.mean(),
                'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
                'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
                'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.,
                'losses/recons_loss': q_recons_loss,
                'info/x_mean_1': 0,
                'info/x_var_1': 0,
                'info/x_mean_2': 0,
                'info/x_var_2': 0,
                'info/ocupied_voxel_pred': (
                        voxel_reconstructed[0, -1] >= 0.5).sum() if voxel_reconstructed != None else 0.,
                'info/ocupied_voxel': tar_voxel_grid[0, -1].sum(),
                'info/scene_diff_occup_wThreshold': voxel_grid[0, -1].sum() - (
                        voxel_reconstructed[0, -1] >= 0.5).sum() if voxel_reconstructed != None else 0.,
                'info/scene_diff_occup': tar_voxel_grid[0, -1].sum() - (
                    voxel_reconstructed[0, -1]).sum() if voxel_reconstructed != None else 0.,
                'info/scene_diff_RGB': tar_voxel_grid[0, 3:6].abs().sum() - voxel_reconstructed[0,
                                                                            0:3].abs().sum() if voxel_reconstructed != None else 0.,
            }

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        ## change
        # if voxel_reconstructed!=None:
        #     self._vis_voxel_grid_reconstructed = torch.cat((voxel_grid[0,:3], #xyz
        #                                     voxel_reconstructed[0,:3], #rgb
        #                                     voxel_grid[0,6:-1], #xyz
        #                                     voxel_reconstructed[0,-1:]), dim=0) #occup
        # else:
        #     self._vis_voxel_grid_reconstructed = None
        self._vis_reconstructed = torch.cat((tar_voxel_grid[0, :3],  # xyz
                                             voxel_reconstructed[0, :3],  # rgb
                                             tar_voxel_grid[0, 6:-1],  # xyz
                                             voxel_reconstructed[0, -1:]), dim=0)  # occup
        self._vis_tar_img = tar_voxel_grid[0]

        if voxel_grid_masked != None:
            self._vis_voxel_grid_masked = voxel_grid_masked[0]
        else:
            self._vis_voxel_grid_masked = None

        self._vis_voxel_grid = voxel_grid[0]

        if self.reconstruction3D_pretraining:
            self._vis_translation_qvalue = None
            self._vis_max_coordinate = None
            self._vis_gt_coordinate = None
        else:
            self._vis_translation_qvalue = self._softmax_q_trans(q_trans[0])
            self._vis_max_coordinate = coords[0]
            self._vis_gt_coordinate = action_trans[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        return {
            'total_loss': total_loss,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }

    def generate(self, step: int, replay_sample: dict, replay: ReplayBuffer) -> dict:
        action_trans = replay_sample['trans_action_indicies'][:, self._layer * 3:self._layer * 3 + 3].int()
        action_rot_grip = replay_sample['rot_grip_action_indicies'].int()
        # action_gripper_pose = replay_sample['gripper_pose']
        action_ignore_collisions = replay_sample['ignore_collisions'].int()
        lang_goal_emb = replay_sample['lang_goal_emb'].float()
        lang_token_embs = replay_sample['lang_token_embs'].float()
        prev_layer_voxel_grid = replay_sample.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = replay_sample.get('prev_layer_bounds', None)
        device = self._device

        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)]
            bounds = torch.cat([cp - self._bounds_offset, cp + self._bounds_offset], dim=1)

        proprio = None
        if self._include_low_dim_state:
            proprio = replay_sample['low_dim_state']

        obs, pcd = self._preprocess_inputs(replay_sample)
        # import ipdb; ipdb.set_trace()
        # batch size
        bs = pcd[0].shape[0]

        ## SE(3) augmentation of point clouds and actions
        # if self._transform_augmentation:
        #     action_trans, \
        #     action_rot_grip, \
        #     pcd = apply_se3_augmentation(pcd,
        #                                  action_gripper_pose,
        #                                  action_trans,
        #                                  action_rot_grip,
        #                                  bounds,
        #                                  self._layer,
        #                                  self._transform_augmentation_xyz,
        #                                  self._transform_augmentation_rpy,
        #                                  self._transform_augmentation_rot_resolution,
        #                                  self._voxel_size,
        #                                  self._rotation_resolution,
        #                                  self._device)

        # forward pass
        q_trans, q_rot_grip, \
        q_collision, \
        _, \
        voxel_grid = self._q(obs,
                             proprio,
                             pcd,
                             lang_goal_emb,
                             lang_token_embs,
                             bounds,
                             prev_layer_bounds,
                             prev_layer_voxel_grid,
                             generate=1)

        # argmax to choose best action
        coords, \
        rot_and_grip_indicies, \
        ignore_collision_indicies = self._q.choose_highest_action(q_trans, q_rot_grip, q_collision)

        ############ add to replay bufffer ###########
        obs_dict = {}
        obs_dict['lang_goal_emb'] = lang_goal_emb[0].detach().cpu().numpy()
        obs_dict['lang_token_embs'] = lang_token_embs[0].detach().cpu().numpy()
        obs_dict['low_dim_state'] = torch.zeros_like(replay_sample['low_dim_state'][0]).detach().cpu().numpy()

        # prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': coords.view(bs, -1).tolist()[0],
            'rot_grip_action_indicies': rot_and_grip_indicies.view(bs, -1).tolist()[0],
            # 'gripper_pose': obs_tp1.gripper_pose,
            'ignore_collisions': ignore_collision_indicies[0].detach().cpu().numpy(),
            # 'lang_goal': np.array([description], dtype=object),
            'voxel_reconstructed': voxel_grid[0].detach().cpu().numpy()
        }

        ## incorrent items in here - TODO: change to zeros
        extra_replay_elements = {
            'front_camera_extrinsics': replay_sample['front_camera_extrinsics'][0].detach().cpu().numpy(),
            'front_camera_intrinsics': replay_sample['front_camera_intrinsics'][0].detach().cpu().numpy(),
            'front_point_cloud': replay_sample['front_point_cloud'][0].detach().cpu().numpy(),
            'front_rgb': replay_sample['front_rgb'][0].detach().cpu().numpy(),
            'gripper_pose': replay_sample['gripper_pose'][0].detach().cpu().numpy(),
            'lang_goal': np.array(['0']),
            'left_shoulder_camera_extrinsics': replay_sample['left_shoulder_camera_extrinsics'][
                0].detach().cpu().numpy(),
            'left_shoulder_camera_intrinsics': replay_sample['left_shoulder_camera_intrinsics'][
                0].detach().cpu().numpy(),
            'left_shoulder_point_cloud': replay_sample['left_shoulder_point_cloud'][0].detach().cpu().numpy(),
            'left_shoulder_rgb': replay_sample['left_shoulder_rgb'][0].detach().cpu().numpy(),
            'right_shoulder_camera_extrinsics': replay_sample['right_shoulder_camera_extrinsics'][
                0].detach().cpu().numpy(),
            'right_shoulder_camera_intrinsics': replay_sample['right_shoulder_camera_intrinsics'][
                0].detach().cpu().numpy(),
            'right_shoulder_point_cloud': replay_sample['right_shoulder_point_cloud'][0].detach().cpu().numpy(),
            'right_shoulder_rgb': replay_sample['right_shoulder_rgb'][0].detach().cpu().numpy(),
            'task': '0',
            'wrist_camera_extrinsics': replay_sample['wrist_camera_extrinsics'][0].detach().cpu().numpy(),
            'wrist_camera_intrinsics': replay_sample['wrist_camera_intrinsics'][0].detach().cpu().numpy(),
            'wrist_point_cloud': replay_sample['wrist_point_cloud'][0].detach().cpu().numpy(),
            'wrist_rgb': replay_sample['wrist_rgb'][0].detach().cpu().numpy(),
        }

        others.update(final_obs)
        others.update(obs_dict)
        others.update(extra_replay_elements)

        # for keys, values in others.items():
        #     try: 
        #         print(keys, values.shape)
        #     except:
        #         print(keys, values)

        # import ipdb; ipdb.set_trace()
        replay.add(replay_sample['action'][0].detach().cpu().numpy(),
                   replay_sample['reward'][0].detach().cpu().item(),
                   bool(replay_sample['terminal'][0].detach().cpu().item()),
                   bool(replay_sample['timeout'][0].detach().cpu().item()),
                   **others)

        ########### replay updated ###########s

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # translation one-hot
        action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt_coord = action_trans[b, :].int()
            action_trans_one_hot[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

        # translation loss
        q_trans_flat = q_trans.view(bs, -1)
        action_trans_one_hot_flat = action_trans_one_hot.view(bs, -1)
        q_trans_loss = self._celoss(q_trans_flat, action_trans_one_hot_flat)

        with_rot_and_grip = rot_and_grip_indicies is not None
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(bs):
                gt_rot_grip = action_rot_grip[b, :].int()
                action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                action_grip_one_hot[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat = q_rot_grip[:, 0 * self._num_rotation_classes:1 * self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1 * self._num_rotation_classes:2 * self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2 * self._num_rotation_classes:3 * self._num_rotation_classes]
            q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes:]
            q_ignore_collisions_flat = q_collision

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
            q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
            q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(q_ignore_collisions_flat, action_ignore_collisions_one_hot)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()

        self._optimizer.zero_grad()
        # total_loss.backward()
        # self._optimizer.step()

        self._summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.,
            'info/ocupied_voxel_pred': (voxel_reconstructed[0, -1] >= 0.5).sum(),
            'info/ocupied_voxel': voxel_grid[0, -1].sum(),
        }

        # if self._lr_scheduler:
        #     self._scheduler.step()
        #     self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._vis_translation_qvalue = self._softmax_q_trans(q_trans[0])
        self._vis_max_coordinate = coords[0]
        self._vis_gt_coordinate = action_trans[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        return {
            'total_loss': total_loss,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:

        # import ipdb; ipdb.set_trace()
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = observation.get('prev_layer_bounds', None)
        # shape: 1,1,77
        lang_goal_tokens = observation.get('lang_goal_tokens', None).long()

        # extract CLIP language embs
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        proprio = None

        if self._include_low_dim_state:
            proprio = observation['low_dim_state']

        obs, pcd = self._act_preprocess_inputs(observation)
        print(pcd[0].shape)

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        proprio = proprio[0].to(self._device)
        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = prev_layer_voxel_grid.to(self._device) if prev_layer_voxel_grid is not None else None
        prev_layer_bounds = prev_layer_bounds.to(self._device) if prev_layer_bounds is not None else None

        # inference
        q_trans, \
        q_rot_grip, \
        q_ignore_collisions, \
        vox_grid, \
        voxel_grid_masked, \
        voxel_reconstructed, \
        tar_voxel_grid, \
        vae_mean, vae_var, \
        x_mean, x_var = self._q(obs,
                                proprio,
                                pcd,
                                lang_goal_emb,
                                lang_token_embs,
                                bounds,
                                prev_layer_bounds,
                                prev_layer_voxel_grid,
                                masked_decoding=self.masked_decoding,
                                masking_ratio=self.masking_ratio,
                                masking_type=self.masking_type,
                                input_masking_ratio=self.input_masking_ratio
                                )

        # softmax Q predictions
        q_trans = self._softmax_q_trans(q_trans)
        q_rot_grip = self._softmax_q_rot_grip(q_rot_grip) if q_rot_grip is not None else q_rot_grip
        q_ignore_collisions = self._softmax_ignore_collision(q_ignore_collisions) \
            if q_ignore_collisions is not None else q_ignore_collisions

        # argmax Q predictions
        coords, \
        rot_and_grip_indicies, \
        ignore_collisions = self._q.choose_highest_action(q_trans, q_rot_grip, q_ignore_collisions)

        rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        ignore_collisions_action = ignore_collisions.int() if ignore_collisions is not None else None

        coords = coords.int()
        attention_coordinate = bounds[:, :3] + res * coords + res / 2

        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            'attention_coordinate': attention_coordinate,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }
        info = {
            'voxel_grid_depth%d' % self._layer: vox_grid,
            'q_depth%d' % self._layer: q_trans,
            'voxel_idx_depth%d' % self._layer: coords
        }
        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate = coords[0]
        self._act_qvalues = q_trans[0].detach()
        self._act_reconstructed = torch.cat((vox_grid[0, :3],  # xyz
                                             voxel_reconstructed[0, :3],  # rgb
                                             vox_grid[0, 6:-1],  # xyz
                                             voxel_reconstructed[0, -1:]), dim=0)
        # save the generated images
        recon_img = transforms.ToTensor()(visualise_voxel(
            self._act_reconstructed.cpu().numpy(),
            self._act_qvalues.cpu().numpy(),
            self._act_max_coordinate.cpu().numpy()))
        file_path = os.environ['img_path']
        files = os.listdir(file_path)
        if len(files) == 0:
            f_name = '0.png'
        else:
            idx = sorted(map(int, [i[:-4] for i in files]))
            print(idx)
            f_name = str(idx[-1] + 1) + '.png'
        print(os.path.join(os.environ['img_path'], f_name))
        save_image(recon_img, os.path.join(os.environ['img_path'], f_name))
        return ActResult((coords, rot_grip_action, ignore_collisions_action),
                         observation_elements=observation_elements,
                         info=info)

    def update_summaries(self) -> List[Summary]:
        summaries = [
            ImageSummary('%s/update_qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._vis_voxel_grid.detach().cpu().numpy(),
                             self._vis_translation_qvalue.detach().cpu().numpy() \
                                 if self._vis_translation_qvalue != None else None,
                             self._vis_max_coordinate.detach().cpu().numpy() \
                                 if self._vis_max_coordinate != None else None,
                             self._vis_gt_coordinate.detach().cpu().numpy() \
                                 if self._vis_gt_coordinate != None else None))),
            ImageSummary('%s/reconstructed_img' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._vis_reconstructed.detach().cpu().numpy(),
                             self._vis_translation_qvalue.detach().cpu().numpy(),
                             self._vis_max_coordinate.detach().cpu().numpy(),
                             self._vis_gt_coordinate.detach().cpu().numpy()))),
            ImageSummary('%s/target_img' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._vis_tar_img.detach().cpu().numpy(),
                             self._vis_translation_qvalue.detach().cpu().numpy(),
                             self._vis_max_coordinate.detach().cpu().numpy(),
                             self._vis_gt_coordinate.detach().cpu().numpy())))
        ]

        if self._vis_voxel_grid_masked != None:
            summaries.append(
                ImageSummary('%s/update_qattention_masked' % self._name,
                             transforms.ToTensor()(visualise_voxel(
                                 self._vis_voxel_grid_masked.detach().cpu().numpy(),
                             )))
            )
        # if self._vis_voxel_grid_reconstructed != None:
        #     summaries.append(
        #         ImageSummary('%s/update_qattention_reconstructed' % self._name,
        #                     transforms.ToTensor()(visualise_voxel(
        #                         self._vis_voxel_grid_reconstructed.detach().cpu().numpy(),
        #                     )))
        #                     )

        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))
        for (name, crop) in (self._crop_summary):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0

            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        # for tag, param in self._q.named_parameters():
        #     # assert not torch.isnan(param.grad.abs() <= 1.0).all()
        #     summaries.append(
        #         HistogramSummary('%s/gradient/%s' % (self._name, tag),
        #                          param.grad))
        #     summaries.append(
        #         HistogramSummary('%s/weight/%s' % (self._name, tag),
        #                          param.data))

        return summaries

    def act_summaries(self) -> List[Summary]:
        print(self._act_reconstructed.shape)
        return [
            ImageSummary('%s/act_Qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._act_voxel_grid.cpu().numpy(),
                             self._act_qvalues.cpu().numpy(),
                             self._act_max_coordinate.cpu().numpy()))),
            ImageSummary('%s/act_reconstruct' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._act_reconstructed.cpu().numpy(),
                             self._act_qvalues.cpu().numpy(),
                             self._act_max_coordinate.cpu().numpy()))),
        ]

    def load_weights(self, savedir: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)
        # import ipdb; ipdb.set_trace()
        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        remove_module_suffix = True if '.module' not in list(merged_state_dict.keys())[-1] \
            else False
        for k, v in state_dict.items():
            if not self._training or remove_module_suffix:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict and '_voxelizer' not in k:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def save_weights(self, savedir: str):
        iteration = savedir.split('/')[-1]
        logging.info("saving checkpoint %s" % iteration)
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))
