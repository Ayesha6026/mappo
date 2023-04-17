#!/bin/sh
env="StarCraft2"
map="8m"
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --n_eval_rollout_threads 1 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 20000000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32 --opponent_dir /home/gmoore/Dev/MappoDev/mappo/onpolicy/scripts/results/StarCraft2/8m/rmappo/check/wandb/run-20230331_164614-f3rsdsfk/files/ \
    --model_dir /home/gmoore/Dev/mappo/onpolicy/scripts/results/StarCraft2/8m/first_bot/
done

# python ../train/train_smac.py --env_name StarCraft2 --algorithm_name rmappo --experiment_name check8m --map_name 8m --seed 1 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 400 --num_env_steps 1000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32
