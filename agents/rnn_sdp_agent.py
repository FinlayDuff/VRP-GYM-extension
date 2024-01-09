import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau

import csv
import logging
import os
import time
from copy import deepcopy
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init


class VRPRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:  # Weight matrix for the input-hidden layer
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:  # Weight matrix for the hidden-hidden layer
                init.xavier_uniform_(param.data)
            elif "bias" in name:  # Bias
                param.data.fill_(0)
            elif (
                "output_layer" in name or "value_layer" in name
            ):  # Linear layer weights
                init.xavier_uniform_(param.data)

    def forward(self, state_sequences):
        # Convert list of state sequences to tensor
        state_tensors = [
            torch.tensor(seq, dtype=torch.float, device=self.device).view(
                -1, seq[0].shape[0] * seq[0].shape[1]
            )
            for seq in state_sequences
        ]
        # Pad sequences
        states_padded = rnn_utils.pad_sequence(
            state_tensors, batch_first=True, padding_value=0
        )
        lengths = torch.tensor([len(seq) for seq in state_sequences], dtype=torch.long)
        # Pack padded sequences
        packed_input = rnn_utils.pack_padded_sequence(
            states_padded, lengths, batch_first=True, enforce_sorted=False
        )

        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, len(state_sequences), self.hidden_dim).to(
            self.device
        )
        c0 = torch.zeros(self.num_layers, len(state_sequences), self.hidden_dim).to(
            self.device
        )
        packed_output, (hn, cn) = self.lstm(packed_input, (h0, c0))
        unpacked_output, _ = rnn_utils.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Get the output of the last timestep for each sequence
        last_timestep_idxs = (
            (lengths - 1)
            .view(-1, 1)
            .expand(len(lengths), unpacked_output.size(2))
            .unsqueeze(1)
        )
        last_timesteps = unpacked_output.gather(1, last_timestep_idxs).squeeze(1)

        # Compute action probabilities and state values
        action_probabilities = F.softmax(self.output_layer(last_timesteps), dim=-1)
        state_values = self.value_layer(last_timesteps).squeeze(-1)

        return action_probabilities, state_values


class SDPAgentRNN:
    def __init__(
        self,
        node_dim: int = 2,
        num_features: int = 7,
        hidden_dim: int = 512,
        num_layers: int = 1,
        lr: float = 1e-4,
        gamma: float = 0.99,
        dropout_rate: float = 0.5,
        csv_path: str = "loss_log.csv",
        seed=69,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.csv_path = csv_path
        self.gamma = gamma

        self.model = VRPRNN(
            input_dim=node_dim * num_features,
            hidden_dim=hidden_dim,
            output_dim=node_dim,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.opt, mode="min", factor=0.1, patience=200, min_lr=1e-6, verbose=True
        )

    def train(
        self,
        env,
        episodes: int = 100,
        check_point_dir: str = "./check_points/",
        eval_interval: int = 50,
    ):
        logging.info("Start Training")
        with open(self.csv_path, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Train Reward", "Eval Reward", "Time"])

        start_time = time.time()
        best_loss = float("inf")

        evaluation_rewards = []
        batch_avg_rewards = []

        for episode in range(episodes):
            self.model.train()
            total_rewards = []
            total_log_probs = []
            total_state_values = []

            env.reset()
            done = False
            while not done:
                state_sequences = (
                    env.get_state_sequence()
                )  # Get batched state sequences
                action_probabilities, state_values = self.model(state_sequences)

                action, log_prob = self.select_actions(action_probabilities, env)
                _, reward, done, _ = env.step(action)

                total_rewards.append(
                    torch.tensor(reward, dtype=torch.float, device=self.device)
                )
                total_log_probs.append(log_prob.squeeze().to(self.device))
                total_state_values.append(state_values.squeeze().to(self.device))

            mean_rewards = torch.stack(total_rewards).cumsum(dim=0)[-1].mean(dim=0)
            batch_avg_rewards.append(mean_rewards)

            # Compute discounted rewards (returns)
            discounted_rewards = self.discount_rewards(torch.stack(total_rewards))

            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-10
            )

            # Calculate policy loss and value loss

            policy_loss = (-torch.stack(total_log_probs) * discounted_rewards).mean()

            # This implements a baseline
            # advantages = -(discounted_rewards - state_values.detach().squeeze())
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            # policy_loss = (-torch.stack(total_log_probs) * advantages).mean()
            # value_loss = F.mse_loss(
            #     torch.stack(total_state_values).squeeze(), discounted_rewards
            # )
            # loss = policy_loss + value_loss

            loss = policy_loss

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.opt.step()
            self.scheduler.step(mean_rewards)

            if episode % eval_interval == 0:
                eval_reward = self.evaluate(env)
                evaluation_rewards.append(eval_reward)
                eval_reward = eval_reward.item()
            else:
                eval_reward = np.nan

            if episode % 50 == 0 and episode != 0:
                logging.info(
                    f"Episode {episode} finished - Loss: {loss.item()} - Reward: {mean_rewards.item()}"
                )

            # log training data
            with open(self.csv_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        episode,
                        loss.item(),
                        mean_rewards.item(),
                        eval_reward,
                        time.time() - start_time,
                    ]
                )

            self.save_model(episode=episode, check_point_dir=check_point_dir)
        # self.plot_learning_curve(batch_avg_rewards, evaluation_rewards)

    def select_actions(self, action_probabilities, env, rollout=False):
        # Mask the actions which are not allowed and normalise the probabilities given these missing actions
        mask = torch.from_numpy(env.generate_mask()).float().to(self.device)
        mask = 1 - mask
        masked_prob = action_probabilities * mask

        # Re-normalize the masked probabilities
        normalized_prob = masked_prob / masked_prob.sum()
        # Re-normalize the masked probabilities
        normalized_prob = masked_prob / masked_prob.sum(dim=1, keepdim=True)

        if rollout:
            # If rollout is True, select actions greedily
            actions = torch.argmax(normalized_prob, dim=1).unsqueeze(1)
            log_probs = torch.log(torch.gather(normalized_prob, 1, actions))
        else:
            # If rollout is False, sample actions stochastically
            # Sample action from the normalized, masked probabilities for the whole batch
            m = torch.distributions.Categorical(normalized_prob)
            actions = m.sample().unsqueeze(1)
            log_probs = m.log_prob(actions.squeeze())

        # # Sample actions from these probabilities
        # m = torch.distributions.Categorical(action_probabilities)
        # actions = m.sample().unsqueeze(1)
        # log_probs = m.log_prob(actions.squeeze())

        return actions.cpu().numpy(), log_probs

    def discount_rewards(self, rewards):
        # Compute the discounted rewards (returns) for each time step

        discounted_r = torch.zeros_like(rewards, device=self.device)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add

        return discounted_r

    def save_model(self, episode: int, check_point_dir: str) -> None:
        """
        Saves the model parameters every 50 episodes.

        Args:
            episode (int): Current episode number
            check_point_dir (str): Directory where the checkpoints
                will be stored.
        """
        if not os.path.exists(check_point_dir):
            os.makedirs(check_point_dir)

        if episode % 50 == 0 and episode != 0:
            torch.save(
                self.model.state_dict(),
                check_point_dir + f"model_epoch_{episode}.pt",
            )

    def save_best_model(self, check_point_dir: str) -> None:
        """
        Saves the best model

        Args:
            episode (int): Current episode number
            check_point_dir (str): Directory where the checkpoints
                will be stored.
        """
        if not os.path.exists(check_point_dir):
            os.makedirs(check_point_dir)

        torch.save(
            self.model.state_dict(),
            check_point_dir + f"best_model.pt",
        )

    def step(self, env):
        """
        Plays the environment to completion for
        both the baseline and the current model.

        Resets the environment beforehand.

        Args:
            env (gym.env): Environment to train on

        Returns:
            (Tuple[torch.tensor, torch.tensor, torch.tensor]):
                Tuple of the loss of the current model, log_probability for the
                current model, state_values
        """
        env.reset()

        # Go through graph batch and get loss
        loss, log_prob, state_values = self.model(env)

        return loss, log_prob, state_values

    def evaluate(self, env, num_episodes=10):
        self.model.eval()

        episode_reward = []
        # This chooses greedily
        with torch.no_grad():
            logging.info("Start Evaluation")
            for ep in range(num_episodes):
                env.reset()
                total_rewards = []
                done = False
                while not done:
                    state_sequences = (
                        env.get_state_sequence()
                    )  # Get batched state sequences
                    action_probabilities, state_values = self.model(state_sequences)

                    action, log_prob = self.select_actions(
                        action_probabilities, env, rollout=True
                    )
                    _, reward, done, _ = env.step(action)

                    total_rewards.append(
                        torch.tensor(reward, dtype=torch.float, device=self.device)
                    )

                mean_rewards = torch.stack(total_rewards).cumsum(dim=0)[-1].mean()
                episode_reward.append(mean_rewards.item())

        total_mean_rewards = np.mean(episode_reward)
        logging.info(f"Average Reward: {total_mean_rewards}")
        return total_mean_rewards

    def plot_learning_curve(self, episode_rewards, evaluation_rewards):
        plt.figure(figsize=(12, 6))
        plt.plot(episode_rewards, label="Training Reward")
        plt.plot(
            np.arange(
                0, len(episode_rewards), len(episode_rewards) / len(evaluation_rewards)
            ),
            evaluation_rewards,
            label="Evaluation Reward",
        )
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()
