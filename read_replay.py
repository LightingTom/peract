import numpy as np
import os
import torch

cameras = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
# errors = np.load('logs/test/PERACT_BC/errors', allow_pickle=True)
# for cname in cameras:
#     print(errors['%s_rgb' % cname].requires_grad, errors['tar_%s_rgb' % cname].requires_grad)

# paths = ['replay/open_drawer/PERACT_BC/seed0/%s' % i for i in os.listdir('replay/open_drawer/PERACT_BC/seed0') if '.replay' in i]
# for p in paths:
#     replay = np.load(p, allow_pickle=True)
#     for cname in cameras:
#         if not np.array_equal(replay['tar_%s_rgb' % cname], replay['%s_rgb' % cname]):
#             print('rgb in replay:%s not equal' % p[-10:])
#         if not np.array_equal(replay['tar_%s_point_cloud' % cname], replay['%s_point_cloud' % cname]):
#             print('pcd in replay:%s not equal' % p[-10:])
#     break

# check utils.extract_obs
# ori_root = 'logs/test/PERACT_BC/info/ori/'
# tar_root = 'logs/test/PERACT_BC/info/tar/'
# ori_files = os.listdir(ori_root)
# tar_files = os.listdir(tar_root)
# # print(len(ori_files) == len(tar_files))
# for i in range(len(ori_files)):
#     ori_info = np.load(ori_root + ori_files[i], allow_pickle=True)
#     tar_info = np.load(tar_root + tar_files[i], allow_pickle=True)
#     print('test rgb')
#     for k in ori_info.keys():
#         if 'rgb' not in k:
#             continue
#         if not np.array_equal(ori_info[k], tar_info[k]):
#             print(i)
#     print('test pcd')
#     for k in ori_info.keys():
#         if 'point_cloud' not in k:
#             continue
#         if not np.array_equal(ori_info[k], tar_info[k]):
#             print(i)

# check rlbench.utils.get_stored_demo
ori_root = 'logs/test/PERACT_BC/info/ori_obs/0'
tar_root = 'logs/test/PERACT_BC/info/tar_obs/0'
ori_obs = np.load(ori_root, allow_pickle=True)
tar_obs = np.load(tar_root, allow_pickle=True)
# print(len(ori_obs) == len(tar_obs))
for i in range(len(ori_obs)):
    if not np.array_equal(ori_obs[i].front_rgb, tar_obs[i].front_rgb):
        print(i, 'front')
    if not np.array_equal(ori_obs[i].left_shoulder_rgb, tar_obs[i].left_shoulder_rgb):
        print(i, 'left_shoulder')
    if not np.array_equal(ori_obs[i].right_shoulder_rgb, tar_obs[i].right_shoulder_rgb):
        print(i, 'right_shoulder')
    if not np.array_equal(ori_obs[i].wrist_rgb, tar_obs[i].wrist_rgb):
        print(i, 'wrist')
