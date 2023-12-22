# export PYOPENGL_PLATFORM=osmesa
# export DISPLAY=:0.0
export PERACT_ROOT=$(pwd)
# export PYTHONPATH=/home/ishika/peract_dir:/home/ishika/peract_dir/YARR:/home/ishika/peract_dir/RLBench
# export CUDA_VISIBLE_DEVICES=0

#,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \
# --nodelist ink-titan
#[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons] \

# no need to mask during eval
# srun --gres=gpu:1 --nodelist ink-lucy --time 2-00:00 \
#     python eval.py \
#     rlbench.tasks=[open_drawer] \
#     rlbench.task_name='singletask_10demos_64x64_noAug_noRecons_occupRGB' \
#     rlbench.demo_path=$PERACT_ROOT/data/test \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=25 \
#     framework.eval_type='missing' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False

# train single task
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# srun --gres=gpu:1 --nodelist ink-lucy --time 1-00:00 \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[open_drawer] \
#     rlbench.task_name='single_open_drawer_64x64_noAug_withReconsx100_occupRGB' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=1 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=1 \
#     replay.path=/tmp/replay/single_open_drawer_64x64_noAug_withReconsx100_occupRGB \
#     replay.max_parallel_processes=1 \
#     method.voxel_sizes=[100] \
#     method.voxel_patch_size=5 \
#     method.voxel_patch_stride=5 \
#     method.num_latents=64 \
#     method.latent_dim=64 \
#     method.final_dim=10 \
#     method.transform_augmentation.apply_se3=True \
#     method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#     method.pos_encoding_with_lang=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     framework.training_iterations=200000 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=0 \
#     framework.log_freq=100 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.load_existing_weights=True \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=1 \
#     ddp.master_port='29505'

## multi-task setup ##
# test_tasks: turn_tap,stack_cups,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,place_wine_at_rack_location,sweep_to_dustpan_of_size
# [close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,put_groceries_in_cupboard,push_buttons,place_shape_in_shape_sorter,slide_block_to_color_target]
# ,open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame
# srun --gres=gpu:1 --nodelist ink-ellie --time 3-00:00 \

# # nvidia-smi
# ## right now NO action loss for test tasks
# # srun --gres=gpu:1 --nodelist allegro-adams --time 3-00:00 \
# # srun --gres=gpu:1 --nodelist glamor-ruby --qos general --time 30:00 \
# CUDA_VISIBLE_DEVICES=3 \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[open_drawer,close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,put_groceries_in_cupboard,push_buttons,place_shape_in_shape_sorter,slide_block_to_color_target] \
#     rlbench.task_name='test' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=1 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=1 \
#     replay.path=/home/ishika/peract_dir/peract/replay/replay_data \
#     replay.load_replay_from_disk=True \
#     replay.max_parallel_processes=1 \
#     method.voxel_sizes=[100] \
#     method.voxel_patch_size=5 \
#     method.voxel_patch_stride=5 \
#     method.num_latents=2048 \
#     method.latent_dim=512 \
#     method.final_dim=10 \
#     method.transform_augmentation.apply_se3=True \
#     method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#     method.pos_encoding_with_lang=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=True \
#     method.masking_ratio=0.8 \
#     method.masking_type='patch' \
#     method.input_masking_ratio=0.5 \
#     method.train_with_seen_objects=False \
#     framework.training_iterations=600000 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=0 \
#     framework.log_freq=10 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.load_existing_weights=False \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=1 \
#     ddp.master_port='29526'

# srun --gres=gpu:1 --nodelist ink-ellie --qos general --time 3-00:00 \
#     python train.py \
#     method=BC_LANG \
#     rlbench.tasks=[open_drawer,close_jar] \
#     rlbench.task_name='test' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=1 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=6 \
#     replay.path=/home/ishika/peract_dir/peract/replay/replay_data \
#     replay.load_replay_from_disk=True \
#     replay.max_parallel_processes=32 \
#     framework.training_iterations=600000 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=0 \
#     framework.log_freq=10 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.load_existing_weights=False \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=1 \
#     ddp.master_port='29520'

# generate data for all tasks
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[slide_block_to_target,put_knife_on_chopping_board,put_knife_in_knife_block,push_buttons,put_toilet_roll_on_stand,slide_block_to_color_target,reach_target,reach_and_drag,place_hanger_on_rack,screw_nail,push_button,pour_from_cup_to_cup,play_jenga,press_switch,put_tray_in_oven,scoop_with_spatula,put_plate_in_colored_dish_rack,set_clock_to_time,setup_checkers,remove_cups,plug_charger_in_power_supply,put_umbrella_in_umbrella_stand,place_cups,put_all_groceries_in_cupboard,put_bottle_in_fridge,set_the_table,put_money_in_safe,solve_puzzle,place_wine_at_rack_location,setup_chess,put_books_on_bookshelf,put_shoes_in_box,put_books_at_shelf_location,take_item_out_of_drawer,toilet_seat_down,toilet_seat_up,take_toilet_roll_off_stand,stack_wine,tv_on,take_frame_off_hanger,take_tray_out_of_oven,turn_oven_on,straighten_rope,weighing_scales,take_money_out_safe,unplug_charger,sweep_to_dustpan_of_size,take_umbrella_out_of_umbrella_stand,take_off_weighing_scales,water_plants,take_usb_out_of_computer,take_shoes_out_of_box,insert_onto_square_peg,empty_container,close_door,phone_on_base,basketball_in_hoop,open_oven,hit_ball_with_queue,pick_and_lift,change_channel,close_drawer,empty_dishwasher,block_pyramid,open_fridge,open_wine_bottle,hockey,close_grill,close_laptop_lid,open_door,light_bulb_out,change_clock,pick_up_cup,lamp_off,lamp_on,get_ice_from_fridge,open_grill,beat_the_buzz,close_box,insert_usb_in_computer,meat_off_grill,meat_on_grill,open_box,close_fridge,move_hanger,hang_frame_on_hanger,open_window] \
#     rlbench.task_name='PT_3D_rep_reconst_only' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=20 \
#     rlbench.demo_path=$PERACT_ROOT/data/train_all_20 \
#     replay.batch_size=6 \
#     replay.path=/home/ishika/peract_dir/peract/replay/replay_data_reconstructionPT_87tasks_run2 \
#     replay.load_replay_from_disk=True \
#     replay.max_parallel_processes=16 \
#     method.voxel_sizes=[100] \
#     method.voxel_patch_size=5 \
#     method.voxel_patch_stride=5 \
#     method.num_latents=2048 \
#     method.latent_dim=512 \
#     method.final_dim=10 \
#     method.transform_augmentation.apply_se3=False \
#     method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#     method.pos_encoding_with_lang=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=True \
#     method.masking_ratio=0.8 \
#     method.masking_type='patch' \
#     method.input_masking_ratio=0.9 \
#     method.train_with_seen_objects=False \
#     method.reconstruction3D_pretraining=True \
#     method.no_language=False \
#     framework.training_iterations=600000 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=0 \
#     framework.log_freq=500 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs_multitask_16tasks/ \
#     framework.load_existing_weights=False \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=4 \
#     ddp.master_port='27530' 

## train MT
# CUDA_VISIBLE_DEVICES=8,9  \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[open_drawer,close_jar,light_bulb_in,put_groceries_in_cupboard,place_shape_in_shape_sorter,sweep_to_dustpan,turn_tap,stack_blocks,put_item_in_drawer,stack_cups,open_microwave,put_rubbish_in_bin,take_plate_off_colored_dish_rack,close_microwave,take_lid_off_saucepan,wipe_desk] \
#     rlbench.task_name='PT_3D_rep_reconst_only' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=20 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=6 \
#     replay.path=/home/ishika/peract_dir/peract/replay/replay_data_16tasks_0 \
#     replay.load_replay_from_disk=True \
#     replay.max_parallel_processes=32 \
#     method.voxel_sizes=[100] \
#     method.voxel_patch_size=5 \
#     method.voxel_patch_stride=5 \
#     method.num_latents=2048 \
#     method.latent_dim=512 \
#     method.final_dim=10 \
#     method.transform_augmentation.apply_se3=True \
#     method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#     method.pos_encoding_with_lang=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     method.masking_ratio=0.8 \
#     method.masking_type='patch' \
#     method.input_masking_ratio=0.5 \
#     method.train_with_seen_objects=False \
#     method.reconstruction3D_pretraining=False \
#     method.no_language=False \
#     framework.training_iterations=300000 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=0 \
#     framework.log_freq=500 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs_multitask_16tasks/ \
#     framework.load_existing_weights=True \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=2 \
#     ddp.master_port='29530' &



# CUDA_VISIBLE_DEVICES=2,3  \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[open_drawer,close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,place_cups,put_groceries_in_cupboard,push_buttons,place_shape_in_shape_sorter,slide_block_to_color_target] \
#     rlbench.task_name='PT_3D_rep_no_language' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=10 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=6 \
#     replay.path=/home/ishika/peract_dir/peract/replay/replay_data \
#     replay.load_replay_from_disk=True \
#     replay.max_parallel_processes=32 \
#     method.voxel_sizes=[100] \
#     method.voxel_patch_size=5 \
#     method.voxel_patch_stride=5 \
#     method.num_latents=2048 \
#     method.latent_dim=512 \
#     method.final_dim=10 \
#     method.transform_augmentation.apply_se3=True \
#     method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#     method.pos_encoding_with_lang=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     method.masking_ratio=0.8 \
#     method.masking_type='patch' \
#     method.input_masking_ratio=0.5 \
#     method.train_with_seen_objects=False \
#     method.reconstruction3D_pretraining=False \
#     method.no_language=False \
#     framework.training_iterations=600000 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=0 \
#     framework.log_freq=500 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.load_existing_weights=True \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=2 \
#     ddp.master_port='29531' &


# CUDA_VISIBLE_DEVICES=0  \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[open_drawer,close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,place_cups,put_groceries_in_cupboard,push_buttons,place_shape_in_shape_sorter,slide_block_to_color_target] \
#     rlbench.task_name='test' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=10 \
#     rlbench.demo_path=$PERACT_ROOT/data/train_all_20 \
#     replay.batch_size=1 \
#     replay.path=/home/ishika/peract_dir/peract/replay/test \
#     replay.load_replay_from_disk=False \
#     replay.max_parallel_processes=8 \
#     method.voxel_sizes=[100] \
#     method.voxel_patch_size=5 \
#     method.voxel_patch_stride=5 \
#     method.num_latents=2048 \
#     method.latent_dim=512 \
#     method.final_dim=10 \
#     method.transform_augmentation.apply_se3=True \
#     method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#     method.pos_encoding_with_lang=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     method.masking_ratio=0.8 \
#     method.masking_type='patch' \
#     method.input_masking_ratio=0.5 \
#     method.train_with_seen_objects=False \
#     method.reconstruction3D_pretraining=False \
#     method.no_language=False \
#     framework.training_iterations=600000 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=0 \
#     framework.log_freq=500 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.load_existing_weights=True \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=1 \
#     ddp.master_port='29510'



# # step 2: generate
# PT task
# test_tasks: turn_tap,stack_cups,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,place_wine_at_rack_location,sweep_to_dustpan_of_size
# srun --gres=gpu:1 --nodelist ink-lucy --time 1-00:00 \
    python train.py \
    method=PERACT_BC \
    rlbench.tasks=[open_drawer] \
    rlbench.task_name='open_drawer_test' \
    rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
    rlbench.demos=10 \
    rlbench.demo_path=$PERACT_ROOT/data/train \
    replay.batch_size=4 \
    replay.path=/home/ishika/peract_dir/peract/replay/multitask_10tasks10demos_64x64_noAug_withReconsx100_occupRGB_PTtask \
    replay.max_parallel_processes=8 \
    method.voxel_sizes=[100] \
    method.voxel_patch_size=5 \
    method.voxel_patch_stride=5 \
    method.num_latents=64 \
    method.latent_dim=64 \
    method.final_dim=10 \
    method.transform_augmentation.apply_se3=True \
    method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
    method.pos_encoding_with_lang=True \
    method.decode_occupnacy=True \
    method.decode_rgb=True \
    framework.training_iterations=600000 \
    framework.num_weights_to_keep=60 \
    framework.start_seed=0 \
    framework.log_freq=100 \
    framework.save_freq=1000 \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.load_existing_weights=False \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=1 \
    ddp.master_port='29509'

# new task, check with single demo
# srun --gres=gpu:1 --nodelist ink-ellie --time 1-00:00 \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[place_wine_at_rack_location] \
#     rlbench.task_name='multitask_10tasks10demos_64x64_noAug_withReconsx100_occupRGB_newtask' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=1 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=1 \
#     replay.path=/home/ishika/peract_dir/peract/replay/multitask_10tasks10demos_64x64_noAug_withReconsx100_occupRGB_newtask \
#     replay.max_parallel_processes=8 \
#     method.voxel_sizes=[100] \
#     method.voxel_patch_size=5 \
#     method.voxel_patch_stride=5 \
#     method.num_latents=64 \
#     method.latent_dim=64 \
#     method.final_dim=10 \
#     method.transform_augmentation.apply_se3=True \
#     method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#     method.pos_encoding_with_lang=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     framework.training_iterations=600000 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=0 \
#     framework.log_freq=100 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.load_existing_weights=False \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=1 \
#     ddp.master_port='29501'

# CUDA_VISIBLE_DEVICES=2 \
# srun --gres=gpu:1 --nodelist ink-lucy --time 1-00:00 \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[open_drawer] \
#     rlbench.task_name='single_open_drawer_64x64_withAugfromPT_5demos' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=5 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=1 \
#     replay.path=/tmp/replay/single_open_drawer_64x64_withAugfromPT_5demos \
#     replay.max_parallel_processes=1 \
#     method.voxel_sizes=[100] \
#     method.voxel_patch_size=5 \
#     method.voxel_patch_stride=5 \
#     method.num_latents=64 \
#     method.latent_dim=64 \
#     method.final_dim=10 \
#     method.transform_augmentation.apply_se3=True \
#     method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
#     method.pos_encoding_with_lang=True \
#     framework.training_iterations=500 \
#     framework.num_weights_to_keep=60 \
#     framework.start_seed=1 \
#     framework.log_freq=100 \
#     framework.save_freq=5000 \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.load_existing_weights=False \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     ddp.num_devices=1 \
#     ddp.master_port='29501'
