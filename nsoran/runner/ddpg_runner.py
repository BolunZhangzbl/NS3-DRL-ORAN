# -- Public Imports
import os
import time
import wandb
import threading
import numpy as np
import tensorflow as tf

# -- Private Imports
from nsoran.utils import *
from nsoran.envs.environments import ORANSimEnv
from nsoran.agents.ddpg import BaseAgentDDPG

# -- Global Variables
tf.get_logger().setLevel('ERROR')

# -- Functions

class DDPGRunner:
    def __init__(self, args):
        self.args = args
        self.stop_flag = threading.Event()   # Thread-safe stop signal
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
        print("Creating Env......")
        self.env = ORANSimEnv(args)
        print("Creating DDPG......")
        self.agent = BaseAgentDDPG(args)

        # Logging variables
        self.ep_rewards = []
        self.step_rewards = []
        self.avg_rewards = []
        self.ep_losses = []
        self.step_actor_losses = []
        self.step_critic_losses = []

    def run(self):
        try:
            """Train the DRL agent over multiple episodes."""
            for episode in range(self.args.num_episodes):
                if self.stop_flag.is_set():  # Check stop flag before starting an episode
                    print("Stopping DRL gracefully...")
                    break

                state = self.env.reset() if episode==0 else state
                episode_reward = 0
                episode_loss = 0

                for step in range(self.args.max_step):
                    if self.stop_flag.is_set():  # Check stop flag at every step
                        print("Stopping DRL gracefully in step loop...")
                        return

                    action = self.agent.act(state)
                    print(f"\naction: {action}\n")
                    next_state, reward, _ = self.env.step(action)

                    # Store step reward
                    self.step_rewards.append(reward)

                    # Record experience & train agent
                    self.agent.record((state, action, reward, next_state))
                    actor_loss_tensor, critic_loss_tensor = self.agent.update()
                    actor_loss = actor_loss_tensor.numpy() if isinstance(actor_loss_tensor, tf.Tensor) else actor_loss_tensor
                    critic_loss = critic_loss_tensor.numpy() if isinstance(critic_loss_tensor, tf.Tensor) else critic_loss_tensor
                    self.step_actor_losses.append(actor_loss)
                    self.step_critic_losses.append(critic_loss)

                    # Update target model periodically
                    self.agent.update_target()

                    # Update state & accumulate episode reward/loss
                    state = next_state
                    episode_reward += reward
                    episode_loss += critic_loss

                    print(f"Step/Episode: {step}/{episode}: "
                          f"Step Reward = {reward:.2e}, "
                          f"Step Loss = {critic_loss:.2e}")

                    # Log step details
                    if self.use_wandb:
                        wandb.log({
                            "Episode": episode,
                            "Step": step,
                            "Step Reward": reward,
                            "Step Loss": critic_loss,
                        })

                # Store episode-level metrics
                self.ep_rewards.append(episode_reward)
                self.avg_rewards.append(np.mean(self.step_rewards[-self.args.max_step:]))
                self.ep_losses.append(episode_loss / self.args.max_step)

                # Log episode details
                avg_reward = self.avg_rewards[-1]
                avg_loss = episode_loss / self.args.max_step

                if self.use_wandb:
                    wandb.log({
                        "Episode": episode,
                        "Episode Reward": episode_reward,
                        "Average Reward": avg_reward,
                        "Average Loss": avg_loss,
                    })

                print(f"Episode: {episode}/{self.args.num_episodes}: "
                      f"Total Reward = {episode_reward:.2e}, "
                      f"Avg Reward = {avg_reward:.2e}, "
                      f"Avg Loss = {avg_loss:.2e}")

                # Save the model periodically
                if episode % 50 == 0:
                    self.agent.save_model()

        finally:
            # Save final model and log final results
            print(f"Finalizing DRL process...")
            self.agent.save_model()
            self._save_results()

            # Finish wandb logging
            if self.use_wandb:
                wandb.finish()
            print(f"DRL training completed and data saved.")

    def stop(self):
        """Signal DRL process to stop gracefully"""
        self.stop_flag.set()

    def _set_seeds(self, seed):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

    def _save_results(self):
        """Save training metrics to file."""
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "lists"))
        os.makedirs(file_path, exist_ok=True)
        save_lists(file_path, self.ep_rewards, self.step_rewards, self.avg_rewards, self.ep_losses,
                   self.step_actor_losses, self.step_critic_losses)
