import os

header = """#!/bin/bash

#SBATCH --mail-user=ishikasi@usc.edu
#SBATCH --mail-type=FAIL  
#SBATCH --gres=gpu:1
#SBATCH --nodelist allegro-adams
#SBATCH --qos general
#SBATCH --cpus-per-task 4
#SBATCH --time=3-00:00
#SBATCH --output=/home/ishika/peract_dir/peract/slurm_outputs/%j.out

export PYOPENGL_PLATFORM=osmesa
export DISPLAY=:0.0
export PERACT_ROOT=$(pwd)
export PYTHONPATH=/home/ishika/peract_dir:/home/ishika/peract_dir/YARR:/home/ishika/peract_dir/RLBench
"""


# experiment_names = ["multitask_10tasks10demos_2048x512",
#                     "multitask_10tasks10demos_2048x512_reconst_Occ",
#                     "multitask_10tasks10demos_2048x512_reconst_OccRGB",
#                     "multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_correct",
#                     "multitask_10tasks10demos_2048x512_reconst_OccRGB_seenobjvar",
#                     "multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_seenobjvar",
#                     "multitask_10tasks10demos_2048x512_fewshotvariation",
#                     "multitask_10tasks10demos_2048x512_reconst_OccRGB_fewshotvariation",
#                     "multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_fewshotvariation",
#                     ]

experiment_names = ["singletask_10tasks10demos_2048x512",
                    "multitask_10tasks10demos_2048x512_reconst_Occ",
                    "singletask_10tasks10demos_2048x512_reconst_OccRGB",
                    "multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_correct",
                    "multitask_10tasks10demos_2048x512_reconst_OccRGB_seenobjvar",
                    "multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_seenobjvar",
                    "multitask_10tasks10demos_2048x512_fewshotvariation",
                    "multitask_10tasks10demos_2048x512_reconst_OccRGB_fewshotvariation",
                    "multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_fewshotvariation",
                    ]
# replay_path = ["10_tasks","10_tasks","10_tasks","10_tasks",
#                 "10_tasks_with_variations","10_tasks_with_variations",
#                 "10_tasks_with_variations","10_tasks_with_variations",
#                 "10_tasks_with_variations"]
replay_path = "replay_data_singletask"

# base_tasks = "open_drawer,close_jar,light_bulb_in,put_groceries_in_cupboard,place_shape_in_shape_sorter,sweep_to_dustpan,turn_tap,stack_blocks,put_item_in_drawer,stack_cups,open_microwave,put_rubbish_in_bin,take_plate_off_colored_dish_rack,close_microwave,take_lid_off_saucepan,wipe_desk"
# variation_tasks = "open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame,open_drawer_test_texture_full"
# # base_tasks = "open_drawer"
# task_lists = [base_tasks, 
#               base_tasks, 
#               base_tasks, 
#               base_tasks,
#               base_tasks + ',' + variation_tasks,
#               base_tasks + ',' + variation_tasks,
#               base_tasks + ',' + variation_tasks,
#               base_tasks + ',' + variation_tasks,
#               base_tasks + ',' + variation_tasks,]

decode_occupnacy = [False, True, True, True, True, True, False, True, True]
decode_rgb = [False, False, True, True, True, True, False, True, True]
masking = [False, False, False, True, False, True, False, False, True]
train_with_seen_objects = [False, False, False, False, True, True, False, False, False]

# f = open(f"slurm_scripts/run_batch_.sh", 'w')
# f.write(header)
expt_ids = [2, 0]
task_list = ['open_drawer', 'close_jar', 
            'place_shape_in_shape_sorter', 'open_microwave',
            'put_rubbish_in_bin']


master_port = 20410
device = -1

for task in task_list:
    for i in expt_ids:
        device += 1
        master_port += 1
        run_cmd = f"""
python train.py \\
method=PERACT_BC \\
rlbench.tasks=[{task}] \\
rlbench.task_name='{experiment_names[i]}_{task}' \\
rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \\
rlbench.demos=10 \\
rlbench.demo_path=$PERACT_ROOT/data/train_all_20 \\
replay.batch_size=6 \\
replay.path=$PERACT_ROOT/replay/{replay_path} \\
replay.load_replay_from_disk=True \\
replay.max_parallel_processes=16 \\
method.voxel_sizes=[100] \\
method.voxel_patch_size=5 \\
method.voxel_patch_stride=5 \\
method.num_latents=2048 \\
method.latent_dim=512 \\
method.final_dim=10 \\
method.transform_augmentation.apply_se3=True \\
method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \\
method.pos_encoding_with_lang=True \\
method.decode_occupnacy={decode_occupnacy[i]} \\
method.decode_rgb={decode_rgb[i]} \\
method.masked_decoding={masking[i]} \\
method.masking_ratio=0.8 \\
method.masking_type='patch' \\
method.input_masking_ratio=0.5 \\
method.train_with_seen_objects={train_with_seen_objects[i]} \\
method.reconstruction3D_pretraining=False \\
framework.training_iterations=80000 \\
framework.num_weights_to_keep=60 \\
framework.start_seed=0 \\
framework.log_freq=500 \\
framework.save_freq=10000 \\
framework.logdir=$PERACT_ROOT/logs_singletask/ \\
framework.load_existing_weights=True \\
framework.csv_logging=True \\
framework.tensorboard_logging=True \\
ddp.num_devices=1 \\
ddp.master_port='{master_port}'
"""

        filename = f"{experiment_names[i]}_{task}"
        print(f"slurm_scripts_singletask/{filename}.sbatch")
        with open(f"slurm_scripts_singletask/{filename}.sbatch", 'w') as f:
            f.write(header)
            f.write(run_cmd)
        os.system(f"sbatch slurm_scripts_singletask/{filename}.sbatch")
        # cmd = header + run_cmd 
        # print(filename)
        # os.system(cmd)

#     f.write(run_cmd)
# f.close()

## change log_freq, demos

