#!/bin/bash

# Environment and script configuration
script="train_dqn.py"
use_cuda=True  # Whether to use CUDA for GPU acceleration
use_wandb=False  # Whether to log metrics to Weights & Biases

# ORAN parameters
num_enb=4  # Number of eNBs
ue_per_enb=3  # Number of UEs per eNB
it_period=100  # Interaction Interval in milliseconds
sim_time=30  # Simulation time in seconds

# DRL parameters
max_step=50  # Maximum number of steps per episode
num_episodes=30  # Total number of episodes for training
last_n=10  # Number of last episodes for evaluation
dqn_lr=1e-3  # Learning rate for the DQN network
gamma=0.99  # Discount factor for future rewards
epsilon=1.0  # Initial exploration rate
epsilon_min=0.01  # Minimum exploration rate
epsilon_decay=0.999  # Decay rate for exploration rate
batch_size=128  # Batch size for training
seed=42  # Random seed for reproducibility

# Run the training script
echo "Starting training with the following configuration:"
echo "-----------------------------------------------"
echo "ORAN Parameters:"
echo "  Number of eNBs: ${num_enb}"
echo "  Number of UEs per eNB: ${ue_per_enb}"
echo "  Interaction Interval: ${it_period} ms"
echo "  Simulation Time: ${sim_time} s"
echo "DRL Parameters:"
echo "  Max Steps per Episode: ${max_step}"
echo "  Number of Episodes: ${num_episodes}"
echo "  Last N Episodes for Evaluation: ${last_n}"
echo "  Actor Learning Rate: ${actor_lr}"
echo "  Critic Learning Rate: ${critic_lr}"
echo "  DQN Learning Rate: ${dqn_lr}"
echo "  Discount Factor (Gamma): ${gamma}"
echo "  Initial Exploration Rate (Epsilon): ${epsilon}"
echo "  Minimum Exploration Rate: ${epsilon_min}"
echo "  Exploration Rate Decay: ${epsilon_decay}"
echo "  Batch Size: ${batch_size}"
echo "  Random Seed: ${seed}"
echo "-----------------------------------------------"

# Run the Python script with the specified arguments
CUDA_VISIBLE_DEVICES=0 python ${script} \
    --use_cuda=${use_cuda} \
    --use_wandb=${use_wandb} \
    --num_enb=${num_enb} \
    --ue_per_enb=${ue_per_enb} \
    --it_period=${it_period} \
    --sim_time=${sim_time} \
    --max_step=${max_step} \
    --num_episodes=${num_episodes} \
    --last_n=${last_n} \
    --dqn_lr=${dqn_lr} \
    --gamma=${gamma} \
    --epsilon=${epsilon} \
    --epsilon_min=${epsilon_min} \
    --epsilon_decay=${epsilon_decay} \
    --batch_size=${batch_size} \
    --seed=${seed}

echo "Training completed."
