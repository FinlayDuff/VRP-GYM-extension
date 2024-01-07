import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import csv
import logging
import os
import time
from copy import deepcopy
from typing import Tuple

import numpy as np


class FFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        """
        We define a network with drop out, BatchNormalisation, activation of LeakyRelu.
        Also defined is a value layer for a baseline in REINFORCE.
        Weight initialisation is handled by the kaiming method.
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define the architecture of the network
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Baseline
        self.value_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Iterate through all modules in the network
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Apply He initialization
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")

                # Initialize biases to 0, if they exist
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, env, rollout=False) -> Tuple[float]:
        # Forward pass through the network

        # Init vars
        done = False
        state, load = env.get_state()
        state = torch.tensor(state, dtype=torch.float, device=self.device)

        # Initialise lists of values to be stored in episode
        rewards = []
        log_probs = []
        state_values = []

        while not done:
            # Pushing the state through the layers
            x = state.view(state.shape[0], -1)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            # store state value
            state_value = self.value_layer(x)

            # Return the action probabilities
            x = self.output_layer(x)
            actions = F.softmax(x, dim=-1)

            # Mask the actions which are not allowed and normalise the probabilities given these missing actions
            mask = torch.from_numpy(env.generate_mask()).float().to(self.device)
            mask = 1 - mask
            masked_prob = actions * mask

            # Re-normalize the masked probabilities
            normalized_prob = masked_prob / masked_prob.sum()
            # Re-normalize the masked probabilities
            normalized_prob = masked_prob / masked_prob.sum(dim=1, keepdim=True)

            if rollout:
                # If rollout is True, select actions greedily
                actions = torch.argmax(normalized_prob, dim=1).unsqueeze(1)
                log_prob = torch.log(torch.gather(normalized_prob, 1, actions))
            else:
                # If rollout is False, sample actions stochastically
                # Sample action from the normalized, masked probabilities for the whole batch
                m = torch.distributions.Categorical(normalized_prob)
                actions = m.sample().unsqueeze(1)
                log_prob = m.log_prob(actions.squeeze())

            state, reward, done, _ = env.step(actions.cpu().numpy())

            log_probs.append(log_prob.squeeze().to(self.device))
            rewards.append(torch.tensor(reward, dtype=torch.float, device=self.device))
            state_values.append(state_value.squeeze().to(self.device))

            state, load = env.get_state()
            state = torch.tensor(state, dtype=torch.float, device=self.device)

        return torch.stack(rewards), torch.stack(log_probs), torch.stack(state_values)


class SDPAgentFF:
    def __init__(
        self,
        node_dim: int = 2,
        num_features: int = 7,
        hidden_dim: int = 512,
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
        self.model = FFNetwork(
            input_dim=node_dim * num_features,
            hidden_dim=hidden_dim,
            output_dim=node_dim,
            dropout_rate=dropout_rate,
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
    ):
        """
        THIS IS BASIC REINFORCE
        """

        logging.info("Start Training")
        with open(self.csv_path, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Time"])

        start_time = time.time()
        best_loss = float("inf")

        for episode in range(episodes):
            self.model.train()

            rewards, log_probs, state_values = self.step(env)

            # Compute discounted rewards (returns)
            discounted_rewards = self.discount_rewards(rewards)

            # Calculate advantages (discounted_rewards - baseline)
            advantages = -(discounted_rewards - state_values.detach().squeeze())

            # Standardize
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            # Calculate policy loss
            policy_loss = (-log_probs * advantages).mean()

            # loss betwee state value and rewards
            value_loss = F.mse_loss(state_values.squeeze(), discounted_rewards)

            loss = policy_loss + value_loss

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.scheduler.step(loss)

            # Check if the current loss is better (lower) than the best loss seen so far
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_path = f"{check_point_dir}/best_model.pt"
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(
                    f"Saved new best model at episode {episode} with loss: {best_loss}"
                )

            if episode % 50 == 0 and episode != 0:
                logging.info(f"Episode {episode} finished - Loss: {loss.item()}")

            # log training data
            with open(self.csv_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        episode,
                        loss.item(),
                        time.time() - start_time,
                    ]
                )

            self.save_model(episode=episode, check_point_dir=check_point_dir)

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
        rewards, log_prob, state_values = self.model(env)

        return rewards, log_prob, state_values

    def evaluate(self, env):
        """
        Evalutes the current model on the given environment.

        Args:
            env (gym.env): TSPAgent (or inherited) environment
                to evaluate

        Returns:
            torch.Tensor: Reward (e.g. -cost) of the current model.
        """
        # This turns off the dropout
        self.model.eval()

        # This chooses greedily
        with torch.no_grad():
            rewards, _, _ = self.model(env, rollout=True)

        return rewards
