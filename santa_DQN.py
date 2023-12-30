import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gym_vrp.envs import SantaIRPEnv

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        # Define your neural network here
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size, hidden_size)
        self.target_model = DQN(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(state)
        return np.argmax(act_values[0].detach().numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                        np.amax(self.target_model(torch.from_numpy(next_state).float()).detach().numpy()))
            target_f = self.model(torch.from_numpy(state).float())
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(torch.from_numpy(state).float()))
            loss.backward()
            self.optimizer.step()

    def train(self, env, num_episodes):
        rewards = []
        for e in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            for time in range(500):  # Adjust according to your environment
                action = self.act(torch.from_numpy(state).float())
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    self.update_target_model()
                    break
            self.replay()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            rewards.append(total_reward)
            print(f"Episode: {e+1}/{num_episodes}, Total Reward: {total_reward}")
        return rewards

    def evaluate(self, env, num_episodes):
        total_rewards = 0
        for _ in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            episode_reward = 0
            done = False
            while not done:
                action = np.argmax(self.model(torch.from_numpy(state).float())[0].detach().numpy())
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = np.reshape(next_state, [1, self.state_size])
            total_rewards += episode_reward
        avg_reward = total_rewards / num_episodes
        print(f"Average Reward: {avg_reward}")
        return avg_reward

    def visualize(self, env):
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])
        done = False
        while not done:
            env.render()  # Render the environment
            action = np.argmax(self.model(torch.from_numpy(state).float())[0].detach().numpy())
            next_state, _, done, _ = env.step(action)
            state = np.reshape(next_state, [1, self.state_size])
        env.close()


batch_size = 128
seed = 23
num_nodes = 10
num_epochs = 251

env = SantaIRPEnv(num_nodes=num_nodes, batch_size=batch_size, seed=seed)
state_size = env.observation_space.shape[0]
print(f"state_size is: {state_size}")
action_size = env.action_space.n
print(f"action_size is: {action_size}")
agent = DQNAgent(state_size, action_size)

# Train the agent
agent.train(env, num_episodes=1000)

# Evaluate the agent
agent.evaluate(env, num_episodes=100)

# Visualise
agent.visualize(env)
