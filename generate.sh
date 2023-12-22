# CUDA_VISIBLE_DEVICES=0 \
    python eval.py \
    rlbench.tasks=[open_drawer] \
    rlbench.task_name='test_recon_noisy_70k' \
    rlbench.demo_path=$PERACT_ROOT/data/ \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=1 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=10 \
    framework.eval_type='last' \
    rlbench.headless=True \
    method.decode_occupnacy=True \
    method.decode_rgb=True \
    method.masked_decoding=False \
    rlbench.headless=True