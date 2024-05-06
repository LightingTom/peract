
    python eval.py \
    rlbench.tasks=[open_drawer] \
    rlbench.task_name='vae_final_enc' \
    rlbench.demo_path=$PERACT_ROOT/data/val \
    framework.logdir=$PERACT_ROOT/logs/ \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_envs=4 \
    framework.start_seed=0 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=25 \
    framework.eval_type=50000 \
    rlbench.headless=True