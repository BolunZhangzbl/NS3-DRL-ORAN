#!/bin/bash

# Environment and script configuration
script="train_dqn.py"
use_cuda=True  # Whether to use CUDA for GPU acceleration
use_wandb=False  # Whether to log metrics to Weights & Biases

# Only add --use_wandb if the variable is True
CUDA_ENABLED=""
if [ "$use_cuda" = true ] ; then
    CUDA_ENABLED="--use_cuda"
fi

# Only add --use_wandb if the variable is True
WANDB_ENABLED=""
if [ "$use_wandb" = true ] ; then
    WANDB_ENABLED="--use_wandb"
fi

# ORAN parameters
num_enb=4  # Number of eNBs
ue_per_enb=3  # Number of UEs per eNB
it_period=100  # Interaction Interval in milliseconds
sim_time=3  # Simulation time in seconds

# DRL parameters
max_step=100  # Maximum number of steps per episode
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
echo "  DQN Learning Rate: ${dqn_lr}"
echo "  Discount Factor (Gamma): ${gamma}"
echo "  Initial Exploration Rate (Epsilon): ${epsilon}"
echo "  Minimum Exploration Rate: ${epsilon_min}"
echo "  Exploration Rate Decay: ${epsilon_decay}"
echo "  Batch Size: ${batch_size}"
echo "  Random Seed: ${seed}"
echo "-----------------------------------------------"

# Run the Python script with the specified arguments
if [ "$use_cuda" = true ]; then
    echo "Using CUDA..."
    CUDA_VISIBLE_DEVICES=0 python ${script} \
        ${CUDA_ENABLED} \
        ${WANDB_ENABLED} \
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
else
    echo "Running on CPU..."
    CUDA_VISIBLE_DEVICES="" python ${script} \
        ${WANDB_ENABLED} \
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
fi

echo "Training completed."
