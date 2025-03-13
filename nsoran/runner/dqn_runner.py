# -- Public Imports
import os
import wandb
import numpy as np
import tensorflow as tf

# -- Private Imports
from nsoran.utils import *
from nsoran.envs.environments import ORANSimEnv
from nsoran.agents.dqn import BaseAgentDQN

# -- Global Variables
tf.get_logger().setLevel('ERROR')

# -- Functions

class DQNRunner:
    def __init__(self, args):
        self.args = args
        self._set_seeds(args.seed)

        # Initialize Weights & Biases (wandb)
        self.use_wandb = args.use_wandb
        if self.use_wandb:
            wandb.init(
                project="NS3-DRL-ORAN",
                name="NS3-DRL-ORAN",
                config=args
            )

        # Initialize environment and agent
        self.env = ORANSimEnv(args)
        self.agent = BaseAgentDQN(args)

        # Logging variables
        self.ep_rewards = []
        self.step_rewards = []
        self.avg_rewards = []
        self.ep_losses = []
        self.step_losses = []

    def run(self):
        """Train the DRL agent over multiple episodes."""
        for episode in range(1, self.args.num_episodes + 1):
            episode_reward, episode_loss = self._run_episode(episode)

            # Log the episode metrics
            self._log_episode(episode, episode_reward, episode_loss)

            # Save the model periodically
            if episode % 10 == 0:
                self.agent.save_model()

        # Save final model and log final results
        self.agent.save_model()
        self._save_results()

        # Finish wandb logging
        if self.use_wandb:
            wandb.finish()

    def _set_seeds(self, seed):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

    def _run_episode(self, episode):
        """Run a single episode of training."""
        state = self.env.reset()
        episode_reward = 0
        episode_loss = 0

        for step in range(1, self.args.max_step + 1):
            action, action_idx = self.agent.act(state)
            next_state, reward, done = self.env.step(action)

            # Store step reward
            self.step_rewards.append(reward)

            # Record experience & train agent
            self.agent.record((state, action_idx, reward, next_state))
            loss = self.agent.update().numpy()
            self.step_losses.append(loss)

            # Update target model periodically
            self.agent.update_target()

            # Update state & accumulate episode reward/loss
            state = next_state
            episode_reward += reward
            episode_loss += loss

            # Log step details
            self._log_step(episode, step, reward, loss)

            if done:
                break

        # Store episode-level metrics
        self.ep_rewards.append(episode_reward)
        self.avg_rewards.append(np.mean(self.step_rewards[-self.args.max_step:]))
        self.ep_losses.append(episode_loss / step)

        return episode_reward, episode_loss

    def _log_step(self, episode, step, reward, loss):
        """Log step details to wandb."""
        if self.use_wandb:
            wandb.log({
                "Episode": episode,
                "Step": step,
                "Step Reward": reward,
                "Step Loss": loss,
            })

    def _log_episode(self, episode, episode_reward, episode_loss):
        """Log episode details and print summary."""
        avg_reward = self.avg_rewards[-1]
        avg_loss = episode_loss / self.args.max_step

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "Episode": episode,
                "Episode Reward": episode_reward,
                "Average Reward": avg_reward,
                "Average Loss": avg_loss,
            })

        # Print summary
        print(f"Episode {episode}/{self.args.num_episodes}: "
              f"Total Reward = {episode_reward:.2e}, "
              f"Avg Reward = {avg_reward:.2e}, "
              f"Avg Loss = {avg_loss:.2e}")

    def _save_results(self):
        """Save training metrics to file."""
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "lists"))
        save_lists(file_path, self.ep_rewards, self.step_rewards, self.avg_rewards, self.ep_losses, self.step_losses)
