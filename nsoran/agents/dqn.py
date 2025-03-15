# -- Public Imports
import os
import gym
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten
from tensorflow.keras.models import Model

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
        self.state_space = args.num_enb*4
        self.action_space = args.num_enb
        self.action_mapper = ActionMapper(minVal=0, maxVal=self.action_space)

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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_func = tf.keras.losses.Huber()
        # self.loss_func = tf.keras.losses.MeanSquaredError()

        # Create Deep Q Network
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        print(self.model.summary())

    def create_model(self):
        input_shape = (self.state_space, 1)

        # Input Layer
        X_input = Input(input_shape)

        # 1D Convolutional Layer
        X = Conv1D(filters=32, kernel_size=3, activation="relu")(X_input)

        # Flatten the output of the CNN to feed into Dense layers
        X = Flatten()(X)

        # Dense Layers
        X = Dense(512, activation="relu")(X)
        X = Dense(512, activation="relu")(X)
        X = Dense(256, activation="relu")(X)

        # Output Layer (action space size)
        output = Dense(self.action_space, activation="linear")(X)

        # Create Model
        model = Model(inputs=X_input, outputs=output)

        return model

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
            q_vals_dist = self.model.predict(state, verbose=0)[0]
            action_idx = tf.argmax(q_vals_dist).numpy()

        action = self.action_mapper.idx_to_bool_action(action_idx)
        return action, action_idx

    def sample(self):
        sample_indices = np.random.choice(min(self.buffer_counter, self.buffer_capacity), self.batch_size)

        state_sample = tf.convert_to_tensor(self.state_buffer[sample_indices])
        action_sample = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_sample = tf.cast(tf.convert_to_tensor(self.reward_buffer[sample_indices]), dtype=tf.float32)
        next_state_sample = tf.convert_to_tensor(self.next_state_buffer[sample_indices])

        return state_sample, action_sample, reward_sample, next_state_sample

    @tf.function
    def update(self):
        state_sample, action_sample, reward_sample, next_state_sample = self.sample()
        action_sample_int = tf.cast(tf.squeeze(action_sample), tf.int32)

        target_q_vals = tf.reduce_max(self.target_model(next_state_sample), axis=1)
        y = reward_sample + self.gamma * target_q_vals
        mask = tf.one_hot(action_sample_int, self.action_space)

        with tf.GradientTape() as tape:
            q_vals = self.model(state_sample)

            q_action = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)

            loss = self.loss_func(y, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    @tf.function
    def update_target(self, tau=0.001):
        # Update target actor
        # self.target_model.set_weights(self.model.get_weights())
        for (a, b) in zip(self.target_model.variables, self.model.variables):
            a.assign(b * tau + (1 - tau))

    def save_model(self, filename="model_dqn.keras"):
        """Save the full model (architecture + weights + optimizer state)."""
        file_path = os.path.join("results", "models", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            self.model.save(file_path)
            print(f"Model saved successfully at {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename="model_dqn.keras"):
        """Load the full model from file."""
        file_path = os.path.join("results", "models", filename)

        try:
            self.model = tf.keras.models.load_model(file_path)
            self.target_model = tf.keras.models.load_model(file_path)
            print(f"Model loaded successfully from {file_path}")
        except Exception as e:
            print(f"Error loading model: {e}")


# def test_save_load_model():
#     # Step 1: Setup mock arguments for DQN Agent
#     class Args:
#         def __init__(self):
#             self.num_enb = 4  # Just a mock value
#             self.batch_size = 64
#             self.epsilon = 0.1
#             self.epsilon_min = 0.01
#             self.epsilon_decay = 0.99
#             self.gamma = 0.99
#             self.dqn_lr = 1e-3
#             self.buffer_capacity = 1000
#
#     # Step 2: Create an instance of the BaseAgentDQN with mock arguments
#     args = Args()
#     agent = BaseAgentDQN(args)
#
#     # Step 3: Save the model
#     agent.save_model()
#
#     # Step 4: Load the model into a new agent instance
#     new_agent = BaseAgentDQN(args)  # Create a new agent
#     new_agent.load_model()
#
#     # Step 5: Verify that the model architecture and weights are the same
#     assert agent.model.summary() == new_agent.model.summary(), "Model architectures do not match!"
#
#     # Verify that model weights are the same
#     original_weights = agent.model.get_weights()
#     loaded_weights = new_agent.model.get_weights()
#
#     for orig, loaded in zip(original_weights, loaded_weights):
#         assert np.array_equal(orig, loaded), "Model weights do not match!"
#
#     print("Test passed: Model saved and loaded successfully.")
#
#
# # Step 6: Run the test function
# if __name__ == "__main__":
#     test_save_load_model()
