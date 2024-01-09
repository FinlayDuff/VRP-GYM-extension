"""
Reproduction Script for all results presented
"""
import csv
from argparse import ArgumentParser
from copy import deepcopy

import torch

from agents import SDPAgentFF, SDPAgentRNN, RandomAgent
from gym_vrp.envs import SantaIRPEnv, SantaIRPEnv_RNN

seed = 123
num_nodes = [3, 5, 7, 10, 15]
batch_size = 128
max_history_length = 10

num_epochs = 1001
lr = 1e-4
gamma = 0.99
dropout_rate = 0.5
hidden_dim_ff = 1024
hidden_dim_rnn = 512
num_layers = 1

csv_path = "evaluation_reproduction.csv"

with open(csv_path, "w+", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Number of Nodes", "Mean Reward"])

for node in num_nodes:
    print(f"Number of Nodes - {node}")
    env_santa_ff = SantaIRPEnv(
        num_nodes=node, batch_size=batch_size, seed=seed, num_draw=3
    )

    env_santa_ff.enable_video_capturing(
        video_save_path=f"./videos/video_test_santa_ff_{node}_{seed}.mp4"
    )

    env_santa_r = deepcopy(env_santa_ff)

    env_santa_rnn = SantaIRPEnv_RNN(
        num_nodes=node,
        batch_size=batch_size,
        seed=seed,
        num_draw=3,
        max_history_length=max_history_length,
    )
    env_santa_rnn.enable_video_capturing(
        video_save_path=f"./videos/video_test_santa_rnn_{node}_{seed}.mp4"
    )

    print(f"\tEvaluating FF")
    agent_santa_ff = SDPAgentFF(
        node_dim=node,
        hidden_dim=hidden_dim_ff,
        lr=lr,
        gamma=gamma,
        dropout_rate=dropout_rate,
        seed=seed,
        csv_path=f"./train_logs/loss_log_santa_ff_{node}_{seed}.csv",
    )

    # load the model in
    agent_santa_ff.model.load_state_dict(
        torch.load(
            f"./check_points/santa_ff_{node}_{seed}/model_epoch_{num_epochs-1}.pt"
        )
    )
    reward_ff = agent_santa_ff.evaluate(
        env_santa_ff,
        num_episodes=num_epochs,
    )

    # Close the video recorder
    env_santa_ff.vid.close()

    print(f"\tEvaluating RNN")
    agent_santa_rnn = SDPAgentRNN(
        node_dim=node,
        num_features=7,
        hidden_dim=hidden_dim_rnn,
        lr=lr,
        gamma=gamma,
        dropout_rate=dropout_rate,
        csv_path=f"./train_logs/loss_log_santa_rnn_{node}_{seed}.csv",
        seed=seed,
        num_layers=num_layers,
    )

    # load the model in
    agent_santa_rnn.model.load_state_dict(
        torch.load(
            f"./check_points/santa_rnn_{num_nodes}_{seed}/model_epoch_{num_epochs-1}.pt"
        )
    )
    reward_rnn = agent_santa_rnn.evaluate(
        env_santa_rnn,
        num_episodes=num_epochs,
    )

    env_santa_rnn.vid.close()

    random_agent = RandomAgent(seed=seed)
    reward_r = random_agent.eval()

    with open(csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([f"FF-Agent", node, reward_ff])
        writer.writerow([f"RNN-Agent", node, reward_rnn])
        writer.writerow([f"Random-Agent", node, reward_r])
