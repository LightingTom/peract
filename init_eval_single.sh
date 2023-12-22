export PYOPENGL_PLATFORM=osmesa
export DISPLAY=:0.0
export PERACT_ROOT=$(pwd)
export PYTHONPATH=/home/ishika/peract_dir:/home/ishika/peract_dir/YARR:/home/ishika/peract_dir/RLBench

export DEMO_PATH=/home/ishika/peract_dir/peract/data/test_50
export eval_log=last_2
export eval_dir=eval_test50
export eval_type="last_2"
export eval_episodes=50

## test script
# export device=6
# export eval_episodes=10
# export task=place_shape_in_shape_sorter
# task_list=[$task,${task}_test_large,${task}_test_small]
# declare -a array=(singletask_10tasks10demos_2048x512_${task}
#                     singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
#                     )
# for ((idx=0; idx<${#array[@]}; ++idx)); do
# echo ${array[idx]}
# CUDA_VISIBLE_DEVICES=$device \
#     python eval.py \
#     rlbench.tasks=$task_list \
#     rlbench.task_name=${array[idx]} \
#     rlbench.demo_path=$DEMO_PATH \
#     framework.logdir=$PERACT_ROOT/logs_singletask/ \
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
#     > $PERACT_ROOT/logs_singletask/$eval_dir/${array[idx]}_$eval_log.txt &
# done


# ## run all evals
export device=0
export task=open_drawer
task_list=[$task,${task}_test_large,${task}_test_small,${task}_test_color_frame,${task}_test_color_full,${task}_test_shape_large_handles,${task}_test_shape_square_handles,${task}_test_texture_frame,${task}_test_texture_full]
declare -a array=(
                    # singletask_10tasks10demos_2048x512_${task}
                    # singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    PT_singletask_10tasks10demos_2048x512_${task}
                    PT_singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    )
for ((idx=0; idx<${#array[@]}; ++idx)); do
echo $device:${array[idx]}
CUDA_VISIBLE_DEVICES=$device \
    python eval.py \
    rlbench.tasks=$task_list \
    rlbench.task_name=${array[idx]} \
    rlbench.demo_path=$DEMO_PATH \
    framework.logdir=$PERACT_ROOT/logs_singletask/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=$eval_episodes \
    framework.eval_type=$eval_type \
    rlbench.headless=True \
    method.decode_occupnacy=False \
    method.decode_rgb=False \
    method.masked_decoding=False \
    > $PERACT_ROOT/logs_singletask/$eval_dir/${array[idx]}_$eval_log.txt &
done



export device=0
export task=close_jar
task_list=[$task,${task}_test_large,${task}_test_small]
declare -a array=(
                    # singletask_10tasks10demos_2048x512_${task}
                    # singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    PT_singletask_10tasks10demos_2048x512_${task}
                    PT_singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    )
for ((idx=0; idx<${#array[@]}; ++idx)); do
echo $device:${array[idx]}
CUDA_VISIBLE_DEVICES=$device \
    python eval.py \
    rlbench.tasks=$task_list \
    rlbench.task_name=${array[idx]} \
    rlbench.demo_path=$DEMO_PATH \
    framework.logdir=$PERACT_ROOT/logs_singletask/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=$eval_episodes \
    framework.eval_type=$eval_type \
    rlbench.headless=True \
    method.decode_occupnacy=False \
    method.decode_rgb=False \
    method.masked_decoding=False \
    > $PERACT_ROOT/logs_singletask/$eval_dir/${array[idx]}_$eval_log.txt &
done


export device=0
export task=open_microwave
task_list=[$task,${task}_test_large,${task}_test_small]
declare -a array=(
                    # singletask_10tasks10demos_2048x512_${task}
                    # singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    PT_singletask_10tasks10demos_2048x512_${task}
                    PT_singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    )
for ((idx=0; idx<${#array[@]}; ++idx)); do
echo $device:${array[idx]}
CUDA_VISIBLE_DEVICES=$device \
    python eval.py \
    rlbench.tasks=$task_list \
    rlbench.task_name=${array[idx]} \
    rlbench.demo_path=$DEMO_PATH \
    framework.logdir=$PERACT_ROOT/logs_singletask/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=$eval_episodes \
    framework.eval_type=$eval_type \
    rlbench.headless=True \
    method.decode_occupnacy=False \
    method.decode_rgb=False \
    method.masked_decoding=False \
    > $PERACT_ROOT/logs_singletask/$eval_dir/${array[idx]}_$eval_log.txt &
done


declare -a device=(0 0)
export task=place_shape_in_shape_sorter
task_list=[$task,${task}_test_large,${task}_test_small]
declare -a array=(
                    # singletask_10tasks10demos_2048x512_${task}
                    # singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    PT_singletask_10tasks10demos_2048x512_${task}
                    PT_singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    )
for ((idx=0; idx<${#array[@]}; ++idx)); do
echo ${device[idx]}:${array[idx]}
CUDA_VISIBLE_DEVICES=${device[idx]} \
    python eval.py \
    rlbench.tasks=$task_list \
    rlbench.task_name=${array[idx]} \
    rlbench.demo_path=$DEMO_PATH \
    framework.logdir=$PERACT_ROOT/logs_singletask/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=$eval_episodes \
    framework.eval_type=$eval_type \
    rlbench.headless=True \
    method.decode_occupnacy=False \
    method.decode_rgb=False \
    method.masked_decoding=False \
    > $PERACT_ROOT/logs_singletask/$eval_dir/${array[idx]}_$eval_log.txt &
done


declare -a device=(5 5)
export task=put_rubbish_in_bin
task_list=[$task,${task}_test_large,${task}_test_small,${task}_test_size]
declare -a array=(
                    # singletask_10tasks10demos_2048x512_${task}
                    # singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    PT_singletask_10tasks10demos_2048x512_${task}
                    PT_singletask_10tasks10demos_2048x512_reconst_OccRGB_${task}
                    )
for ((idx=0; idx<${#array[@]}; ++idx)); do
echo ${device[idx]}:${array[idx]}
CUDA_VISIBLE_DEVICES=${device[idx]} \
    python eval.py \
    rlbench.tasks=$task_list \
    rlbench.task_name=${array[idx]} \
    rlbench.demo_path=$DEMO_PATH \
    framework.logdir=$PERACT_ROOT/logs_singletask/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=$eval_episodes \
    framework.eval_type=$eval_type \
    rlbench.headless=True \
    method.decode_occupnacy=False \
    method.decode_rgb=False \
    method.masked_decoding=False \
    > $PERACT_ROOT/logs_singletask/$eval_dir/${array[idx]}_NOT_$eval_log.txt &
done


