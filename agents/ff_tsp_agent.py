import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import logging
import os
import time
from copy import deepcopy
from typing import Tuple

import numpy as np


class TSPFFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define the architecture of the network
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, env, rollout=False) -> Tuple[float]:
        # Forward pass through the network

        done = False
        irp_state, irp_load = env.get_state()
        state = torch.tensor(irp_state, dtype=torch.float, device=self.device)
        rewards = []
        log_probs = []

        while not done:
            # Pushing the state through the layers
            x = state.view(state.shape[0], -1)
            print(1)
            print(x.shape)
            print(x)
            x = self.layer1(x)
            print(2)
            print(x)
            x = self.layer2(x)
            print(3)
            print(x)
            x = self.output_layer(x)
            print(4)
            print(x)
            actions = F.softmax(x, dim=-1)  # Return the action probabilities

            # Mask the actions which are not allowed and normalise teh probabilities given these missing actions
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
                actions = m.sample().unsqueeze(
                    1
                )  # actions should have shape (batch_size, 1)
                log_prob = m.log_prob(actions.squeeze())

            state, loss, done, _ = env.step(actions.cpu().numpy())

            log_probs.append(log_prob.squeeze().to(self.device))
            rewards.append(torch.tensor(loss, dtype=torch.float, device=self.device))
            irp_state, irp_load = env.get_state()
            state = torch.tensor(irp_state, dtype=torch.float, device=self.device)
        return torch.stack(rewards), torch.stack(log_probs)


class TSPAgentFF:
    def __init__(
        self,
        node_dim: int = 2,
        hidden_dim: int = 512,
        lr: float = 1e-4,
        gamma: float = 0.99,
        csv_path: str = "loss_log.csv",
        seed=69,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.csv_path = csv_path
        self.gamma = gamma
        self.model = TSPFFNetwork(
            input_dim=node_dim * 4,
            hidden_dim=hidden_dim,
            output_dim=node_dim,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

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

        for episode in range(episodes):
            self.model.train()

            rewards, log_probs = self.step(env, [False, False])

            # Compute discounted rewards (returns)
            discounted_rewards = self.discount_rewards(rewards)

            # Calculate loss
            loss = (-log_probs * discounted_rewards).mean()

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

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
        # Standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_r -= discounted_r.mean()
        discounted_r /= discounted_r.std() + 1e-10
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

    def step(self, env, rollouts: Tuple[bool, bool]):
        """
        Plays the environment to completion for
        both the baseline and the current model.

        Resets the environment beforehand.

        Args:
            env (gym.env): Environment to train on
            rollouts (Tuple[bool, bool]): Each entry decides
                if we sample the actions from the learned
                distribution or act greedy. Indices are for
                the current model (0) and the baseline (1).

        Returns:
            (Tuple[torch.tensor, torch.tensor, torch.tensor]):
                Tuple of the loss of the current model, the loss
                of the baseline and the log_probability for the
                current model.
        """
        env.reset()
        # env_baseline = deepcopy(env)

        # Go through graph batch and get loss
        loss, log_prob = self.model(env)
        # with torch.no_grad():
        #     loss_b, _ = self.target_model(env_baseline, rollouts[0])

        return loss, log_prob

    def evaluate(self, env):
        """
        Evalutes the current model on the given environment.

        Args:
            env (gym.env): TSPAgent (or inherited) environment
                to evaluate

        Returns:
            torch.Tensor: Reward (e.g. -cost) of the current model.
        """
        self.model.eval()

        with torch.no_grad():
            loss, _ = self.model(env, rollout=True)

        return loss
