# -- Public Imports
import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# -- Private Imports
from nsoran.utils import *

# -- Global Variables

dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -- Functions

# Base Agent for DQN
class BaseAgentDQN:
    """
    DQN Agent
    """
    def __init__(self, args):
        self.state_space = args.num_enb * args.num_state
        self.action_space = args.num_action ** args.num_enb
        self.action_mapper = ActionMapper(minVal=0, maxVal=self.action_space-1)

        # Buffer
        self.buffer_counter = 0
        self.buffer_capacity = args.buffer_capacity if hasattr(args, 'buffer_capacity') else int(1e4)
        self.state_buffer = np.zeros((self.buffer_capacity, self.state_space))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_space))

        # Hyper-parameters
        self.batch_size = args.batch_size if hasattr(args, 'batch_size') else 128
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.gamma = args.gamma  # Discount factor
        self.learning_rate = args.dqn_lr  # Learning rate for the DQN network

        # Create Deep Q Network
        device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.model.to(device)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.HuberLoss()
        # self.loss_func = nn.MSELoss()

    def create_model(self):
        class DQNModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(DQNModel, self).__init__()
                
                self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)  # Kernel size = 3
                # Calculate the output length after convolution
                conv_output_dim = input_dim - 2  # Since kernel_size = 3, padding is assumed to be 0
                
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(32 * conv_output_dim, 512)  # Adjust the input dimension based on the conv output
                self.fc2 = nn.Linear(512, 256)
                self.output = nn.Linear(256, output_dim)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.flatten(x)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.output(x)
                return x

        return DQNModel(self.state_space, self.action_space)

    def record(self, obs_tuple):
        assert len(obs_tuple) == 4

        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def act(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        if state.ndim==1:
            state = np.expand_dims(state, axis=0)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(self.action_space)
        else:
            with torch.no_grad():
                q_vals_dist = self.model(torch.FloatTensor(state)).squeeze(0)
                action_idx = torch.argmax(q_vals_dist).item()

        action = self.action_mapper.idx_to_4base_action(action_idx)
        return action, action_idx

    def sample(self):
        sample_indices = np.random.choice(min(self.buffer_counter, self.buffer_capacity), self.batch_size)
        state_sample = torch.FloatTensor(self.state_buffer[sample_indices]).to(self.device)
        action_sample = torch.LongTensor(self.action_buffer[sample_indices]).to(self.device)
        reward_sample = torch.FloatTensor(self.reward_buffer[sample_indices]).to(self.device)
        next_state_sample = torch.FloatTensor(self.next_state_buffer[sample_indices]).to(self.device)

        return state_sample, action_sample, reward_sample, next_state_sample

    def update(self):
        state_sample, action_sample, reward_sample, next_state_sample = self.sample()
        action_sample_int = action_sample.squeeze(1).long()

        with torch.no_grad():
            best_next_actions = torch.argmax(self.model(next_state_sample), dim=1)
            target_q_values = self.target_model(next_state_sample).gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            y = reward_sample.squeeze(1) + self.gamma * target_q_values

        q_vals = self.model(state_sample)
        q_action = q_vals.gather(1, action_sample_int.unsqueeze(1)).squeeze(1)

        loss =self.loss_func(q_action, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self, tau=0.001):
        # Update target actor
        # self.target_model.set_weights(self.model.get_weights())
        for (a, b) in zip(self.target_model.parameters(), self.model.parameters()):
            a.data.copy_(tau * b.data + (1 - tau) * a.data)

    def save_model(self, filename="model_dqn.pth"):
        """Save the model's state dictionary."""
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "models", filename))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, file_path)
            logging.info(f"Model saved successfully at {file_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, filename="model_dqn.pth"):
        """Load the model's state dictionary."""
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "models", filename))

        try:
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f"Model loaded successfully from {file_path}")
        except Exception as e:
            logging.exception(f"Error loading model: {e}")

    def save_state_buffer(self, filename="state_buffer.npz"):
        """Save state buffer"""
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "results", "lists", filename))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            state_data = self.state_buffer[:self.buffer_counter]
            if state_data.shape[1] < 5:
                raise ValueError("State buffer does not have enough columns to save.")

            prbs = state_data[:, 0]  # 1st column
            tp = state_data[:, 1]  # 2nd column
            sinr = state_data[:, 2]  # 3rd column
            tx_power = state_data[:, 3]  # 4th column
            ac = state_data[:, 4]  # 5th column

            np.savez(file_path, prbs=prbs, tp=tp, sinr=sinr, tx_power=tx_power, ac=ac)
            logging.info(f"State Buffer saved successfully at {file_path}")
        except Exception as e:
            logging.exception(f"Error saving State Buffer: {e}")



# def test_save_load_model():
#     class Args:
#         def __init__(self):
#             self.num_enb = 4
#             self.batch_size = 64
#             self.epsilon = 0.1
#             self.epsilon_min = 0.01
#             self.epsilon_decay = 0.99
#             self.gamma = 0.99
#             self.dqn_lr = 1e-3
#             self.buffer_capacity = 1000

#     args = Args()
#     agent = BaseAgentDQN(args)

#     # Save the model
#     agent.save_model()

#     # Load the model into a new agent instance
#     new_agent = BaseAgentDQN(args)
#     new_agent.load_model()

#     # Verify that the model weights are the same
#     for param1, param2 in zip(agent.model.parameters(), new_agent.model.parameters()):
#         assert torch.equal(param1, param2), "Model weights do not match!"

#     logging.info("Test passed: Model saved and loaded successfully.")

# if __name__ == "__main__":
#     test_save_load_model()
