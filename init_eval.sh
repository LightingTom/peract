export PYOPENGL_PLATFORM=osmesa
export DISPLAY=:0.0
export PERACT_ROOT=$(pwd)
export PYTHONPATH=/home/ishika/peract_dir:/home/ishika/peract_dir/YARR:/home/ishika/peract_dir/RLBench

export DEMO_PATH=/home/ishika/peract_dir/peract/data/test_50
# task_list=[open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame]
task_list=[open_drawer,close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,put_groceries_in_cupboard,push_buttons,place_shape_in_shape_sorter,open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame]

#,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \
# --nodelist ink-titan
#[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons] \
export eval_log=test_50_variations_170to_
export eval_dir=eval_test50
export eval_type="170000_to_200000"
export eval_episodes=50

task_list=[close_jar_test_large,place_shape_in_shape_sorter_test_small,close_jar_test_small,open_drawer_test_large,open_drawer_test_small,place_shape_in_shape_sorter_test_large]
# CUDA_VISIBLE_DEVICES=7 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=$eval_type \
#     rlbench.headless=True \
#     method.decode_occupnacy=False \
#     method.decode_rgb=False \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=7 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=$eval_type \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_$eval_log.txt &


# CUDA_VISIBLE_DEVICES=0 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='PT_3D_rep' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=$eval_type \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/PT_3D_rep_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=0 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='PT_3D_rep_no_language' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=$eval_type \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/PT_3D_rep_no_language_$eval_log.txt &


export DEMO_PATH=/home/ishika/peract_dir/peract/data/test_50
# task_list=[open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame]
task_list=[open_drawer,close_jar,light_bulb_in,put_groceries_in_cupboard,place_shape_in_shape_sorter,sweep_to_dustpan,turn_tap,stack_blocks,put_item_in_drawer,stack_cups,open_microwave,put_rubbish_in_bin,take_plate_off_colored_dish_rack,close_microwave,take_lid_off_saucepan,wipe_desk]
#,put_groceries_in_cupboard,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,stack_cups,turn_tap,place_wine_at_rack_location,slide_block_to_color_target,sweep_to_dustpan_of_size] \
# --nodelist ink-titan
#[close_jar,insert_onto_square_peg,light_bulb_in,meat_off_grill,open_drawer,place_cups,place_shape_in_shape_sorter,push_buttons] \
export eval_log=test_50
export eval_dir=eval_val10
export eval_type="last_2"
export eval_episodes=50

CUDA_VISIBLE_DEVICES=0 \
    python eval.py \
    rlbench.tasks=$task_list \
    rlbench.task_name='PT_3D_rep_reconst_only' \
    rlbench.demo_path=$DEMO_PATH \
    framework.logdir=$PERACT_ROOT/logs_multitask_16tasks/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=$eval_episodes \
    framework.eval_type=$eval_type \
    rlbench.headless=True \
    method.decode_occupnacy=True \
    method.decode_rgb=True \
    method.masked_decoding=False \
    > $PERACT_ROOT/logs_multitask_16tasks/$eval_dir/PT_3D_rep_reconst_only_$eval_log.txt &

export eval_log=test_50_variations
task_list=[close_jar_test_large,open_microwave_test_large,place_shape_in_shape_sorter_test_small,put_rubbish_in_bin_test_small,close_jar_test_small,open_microwave_test_small,put_rubbish_in_bin_test_large,open_drawer_test_large,open_drawer_test_small,place_shape_in_shape_sorter_test_large,put_rubbish_in_bin_test_size]

CUDA_VISIBLE_DEVICES=0 \
    python eval.py \
    rlbench.tasks=$task_list \
    rlbench.task_name='PT_3D_rep_reconst_only' \
    rlbench.demo_path=$DEMO_PATH \
    framework.logdir=$PERACT_ROOT/logs_multitask_16tasks/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=$eval_episodes \
    framework.eval_type=$eval_type \
    rlbench.headless=True \
    method.decode_occupnacy=True \
    method.decode_rgb=True \
    method.masked_decoding=False \
    > $PERACT_ROOT/logs_multitask_16tasks/$eval_dir/PT_3D_rep_reconst_only_$eval_log.txt &


# CUDA_VISIBLE_DEVICES=4 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='new_16tasks_multitask_10tasks10demos_2048x512' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask_16tasks/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=$eval_type \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask_16tasks/$eval_dir/new_16tasks_multitask_10tasks10demos_2048x512_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=5 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='new_16tasks_multitask_10tasks10demos_2048x512_reconst_OccRGB' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask_16tasks/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=$eval_type \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask_16tasks/$eval_dir/new_16tasks_multitask_10tasks10demos_2048x512_reconst_OccRGB_$eval_log.txt &


# CUDA_VISIBLE_DEVICES=5 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=$eval_type \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=4 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_correct' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=140000 \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_correct_$eval_log.txt &

# # CUDA_VISIBLE_DEVICES=4 \
# #     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB_seenobjvar' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=$eval_type \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_seenobjvar_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=4 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='PT_3D_rep_no_language' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=40000 \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/PT_3D_rep_no_language_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=4 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_Occ' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=$eval_episodes \
#     framework.eval_type=130000 \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=False \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_Occ_$eval_log.txt &


# CUDA_VISIBLE_DEVICES=9 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=False \
#     method.decode_rgb=False \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=9 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_Occ' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_Occ_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=9 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=9 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_correct' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_correct_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=0 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB_seenobjvar' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_seenobjvar_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=0 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_seenobjvar' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_seenobjvar_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=2 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_fewshotvariation' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=False \
#     method.decode_rgb=False \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_fewshotvariation_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=2 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB_fewshotvariation' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_fewshotvariation_$eval_log.txt &

# CUDA_VISIBLE_DEVICES=3 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_fewshotvariation' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_multitask/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='last' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False \
#     > $PERACT_ROOT/logs_multitask/$eval_dir/multitask_10tasks10demos_2048x512_reconst_OccRGB_patchmask_fewshotvariation_$eval_log.txt &

# ###################################################################################################
###################################################################################################
###################################################################################################

# # no need to mask during eval
# CUDA_VISIBLE_DEVICES=1 \
#     python eval.py \
#     rlbench.tasks=[open_drawer,open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame] \
#     rlbench.task_name='singletask_10demos_64x64_noAug_Recons_occupRGB_withvariationdemos' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='0_to_100000' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False &

# no need to mask during eval
# srun --gres=gpu:1 --nodelist ink-lucy --time 2-00:00 \
#     python eval.py \
#     rlbench.tasks=[open_drawer] \
#     rlbench.task_name='singletask_10demos_64x64_noAug_Recons_occupRGB_masking_with50p' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=25 \
#     framework.eval_type='0_to_100000' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False &

# # no need to mask during eval
# CUDA_VISIBLE_DEVICES=0 \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name='singletask_10demos_64x64_noAug_Recons_occupRGB_masking_with50p' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='0_to_100000' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False &

# # no need to mask during eval
# CUDA_VISIBLE_DEVICES=2 \
#     python eval.py \
#     rlbench.tasks=[open_drawer,open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame] \
#     rlbench.task_name='singletask_10demos_64x64_noAug_Recons_occupRGB_masking_withvariationdemos' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='0_to_100000' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False &

# no need to mask during eval
# srun --gres=gpu:1 --nodelist ink-lucy --time 1-00:00 \
#     python eval.py \
#     rlbench.tasks=[open_drawer,open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame] \
#     rlbench.task_name='singletask_10demos_64x64_noAug_Recons_occupRGB_withvariationdemos_reconslossonly' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='0_to_100000' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False &

# # no need to mask during eval
# srun --gres=gpu:1 --nodelist ink-lucy --time 1-00:00 \
#     python eval.py \
#     rlbench.tasks=[open_drawer,open_drawer_test_color_frame,open_drawer_test_color_full,open_drawer_test_shape_large_handles,open_drawer_test_shape_small_body,open_drawer_test_shape_square_handles,open_drawer_test_texture_frame] \
#     rlbench.task_name='singletask_10demos_64x64_noAug_Recons_occupRGB_masking_withvariationdemos_reconslossonly' \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs/ \
#     framework.csv_logging=True \
#     framework.tensorboard_logging=True \
#     framework.eval_envs=1 \
#     framework.start_seed=0 \
#     framework.eval_from_eps_number=0 \
#     framework.eval_episodes=10 \
#     framework.eval_type='0_to_100000' \
#     rlbench.headless=True \
#     method.decode_occupnacy=True \
#     method.decode_rgb=True \
#     method.masked_decoding=False &

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
# srun --gres=gpu:1 --nodelist ink-ellie --time 3-00:00 \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[open_drawer] \
#     rlbench.task_name='singletask_10demos_64x64_noAug_Recons_occupRGB_patchmasking' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=10 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=1 \
#     replay.path=/home/ishika/peract_dir/peract/replay/singletask_10demos_64x64_noAug_Recons_occupRGB_patchmasking \
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
#     method.masked_decoding=True \
#     method.masking_ratio=0.8 \
#     method.masking_type='patch' \
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
#     ddp.master_port='29520'

# step 2: generate
# PT task
# test_tasks: turn_tap,stack_cups,put_item_in_drawer,put_money_in_safe,reach_and_drag,stack_blocks,place_wine_at_rack_location,sweep_to_dustpan_of_size
# srun --gres=gpu:1 --nodelist ink-lucy --time 1-00:00 \
#     python train.py \
#     method=PERACT_BC \
#     rlbench.tasks=[close_jar] \
#     rlbench.task_name='multitask_10tasks10demos_64x64_noAug_withReconsx100_occupRGB_PTtask' \
#     rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
#     rlbench.demos=1 \
#     rlbench.demo_path=$PERACT_ROOT/data/train \
#     replay.batch_size=1 \
#     replay.path=/home/ishika/peract_dir/peract/replay/multitask_10tasks10demos_64x64_noAug_withReconsx100_occupRGB_PTtask \
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
#     ddp.master_port='29509'

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
