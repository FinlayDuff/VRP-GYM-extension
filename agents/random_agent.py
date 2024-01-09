import torch
import torch.nn as nn
import numpy as np


class RandomAgent(nn.Module):
    """
    Acts randomly within a TSPEnv (or inherited).
    """

    def __init__(self, seed: int = 69):
        super().__init__()
        np.random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, env, num_episodes=10) -> float:
        episode_reward = []
        # This chooses greedily
        for ep in range(num_episodes):
            env.reset()
            total_rewards = []
            done = False
            while not done:
                action = self.select_actions(env)
                _, reward, done, _ = env.step(action)

                total_rewards.append(
                    torch.tensor(reward, dtype=torch.float, device=self.device)
                )

            mean_rewards = torch.stack(total_rewards).cumsum(dim=0)[-1].mean()
            episode_reward.append(mean_rewards.item())

        total_mean_rewards = np.mean(episode_reward)
        return total_mean_rewards

    def select_actions(self, env):
        action_probabilities = self.generate_uniform_probabilities(
            env.batch_size, env.num_nodes
        )
        # Mask the actions which are not allowed and normalise the probabilities given these missing actions
        mask = torch.from_numpy(env.generate_mask()).float().to(self.device)
        mask = 1 - mask
        masked_prob = action_probabilities * mask

        # Re-normalize the masked probabilities
        normalized_prob = masked_prob / masked_prob.sum()
        # Re-normalize the masked probabilities
        normalized_prob = masked_prob / masked_prob.sum(dim=1, keepdim=True)

        m = torch.distributions.Categorical(normalized_prob)
        actions = m.sample().unsqueeze(1)

        return actions.cpu().numpy()

    def generate_uniform_probabilities(self, batch_size, num_nodes):
        probabilities = np.ones((batch_size, num_nodes)) / num_nodes
        return torch.from_numpy(probabilities).float().to(self.device)
