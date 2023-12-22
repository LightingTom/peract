# export CUDA_VISIBLE_DEVICE=0

    python train.py \
    method.name=PERACT_BC \
    rlbench.tasks=[open_drawer] \
    rlbench.task_name='fct5_1218' \
    rlbench.cameras=[front,left_shoulder,right_shoulder,wrist] \
    rlbench.demos=10 \
    rlbench.demo_path=$PERACT_ROOT/data/ \
    replay.batch_size=4 \
    replay.path=$PERACT_ROOT/replay/ \
    replay.max_parallel_processes=16 \
    replay.load_replay_from_disk=False \
    method.voxel_sizes=[100] \
    method.masked_decoding=False \
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
    method.factor=5 \
    framework.training_iterations=100001 \
    framework.num_weights_to_keep=60 \
    framework.start_seed=0 \
    framework.log_freq=2000 \
    framework.save_freq=2000 \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.load_existing_weights=False \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    ddp.num_devices=1 \
    ddp.master_port='29511'

