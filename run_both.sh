#!/bin/bash

# Environment and script configuration
script="run_both.py"

# Default values (as per the parser defaults)
use_cuda=false    # Whether to use CUDA for GPU acceleration (default: disabled)
use_wandb=false   # Whether to log metrics to Weights & Biases (default: disabled)
stream_ns3=true   # Whether to enable stream_ns3 (default: enabled)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --use_cuda)
            use_cuda=true  # Enable CUDA if flag is provided
            shift
            ;;
        --use_wandb)
            use_wandb=true  # Enable Weights & Biases logging
            shift
            ;;
        --no-stream_ns3)
            stream_ns3=false  # Allow disabling stream_ns3
            shift
            ;;
        --stream_ns3)
            stream_ns3=true  # Explicitly enable stream_ns3
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Conditional flag settings based on parsed arguments
CUDA_ENABLED=""
if [ "$use_cuda" = true ]; then
    CUDA_ENABLED="--use_cuda"
fi

WANDB_ENABLED=""
if [ "$use_wandb" = true ]; then
    WANDB_ENABLED="--use_wandb"
fi

STREAM_NS3_ENABLED=""
if [ "$stream_ns3" = true ]; then
    STREAM_NS3_ENABLED="--stream_ns3"
fi

# ORAN parameters
num_enb=4          # Number of eNBs
ue_per_enb=3       # Number of UEs per eNB
it_period=100      # Interaction Interval in milliseconds
sim_time=300      # Simulation time in seconds

# DRL parameters
max_step=100       # Maximum number of steps per episode
num_episodes=30    # Total number of episodes for training
last_n=10          # Number of last episodes for evaluation
dqn_lr=1e-3        # Learning rate for the DQN network
gamma=0.99         # Discount factor for future rewards
epsilon=1.0        # Initial exploration rate
epsilon_min=0.01   # Minimum exploration rate
epsilon_decay=0.999  # Decay rate for exploration rate
batch_size=128     # Batch size for training
seed=42            # Random seed for reproducibility

# Echo the configuration
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
        ${STREAM_NS3_ENABLED} \
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
        ${STREAM_NS3_ENABLED} \
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
