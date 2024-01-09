from gym_vrp.envs import SantaIRPEnv, SantaIRPEnv_RNN
from agents import SDPAgentRNN, SDPAgentFF

seed = 123
num_nodes = [3, 5, 7, 10, 15, 20, 30]
batch_size = 128
max_history_length = 10

num_epochs = 501
lr = 1e-3
gamma = 0.99
dropout_rate = 0.5
hidden_dim_ff = 1024
hidden_dim_rnn = 512
num_layers = 2


for node in num_nodes:
    print(f"number of nodes: {node}")
    env_santa_ff = SantaIRPEnv(
        num_nodes=node, batch_size=batch_size, seed=seed, num_draw=3
    )
    env_santa_rnn = SantaIRPEnv_RNN(
        num_nodes=node,
        batch_size=batch_size,
        seed=seed,
        num_draw=3,
        max_history_length=max_history_length,
    )

    print(f"\tTraining FF")
    agent_santa_ff = SDPAgentFF(
        node_dim=node,
        hidden_dim=hidden_dim_ff,
        lr=lr,
        gamma=gamma,
        dropout_rate=dropout_rate,
        seed=seed,
        csv_path=f"./train_logs/loss_log_santa_ff_{node}_{seed}.csv",
    )
    agent_santa_ff.train(
        env_santa_ff,
        episodes=num_epochs,
        check_point_dir=f"./check_points/santa_ff_{node}_{seed}/",
    )

    # print(f"\tTraining RNN")
    # agent = SDPAgentRNN(
    #     node_dim=node,
    #     num_features=7,
    #     hidden_dim=hidden_dim_rnn,
    #     lr=lr,
    #     gamma=gamma,
    #     dropout_rate=dropout_rate,
    #     csv_path=f"./train_logs/loss_log_santa_rnn_{node}_{seed}.csv",
    #     seed=seed,
    #     num_layers=num_layers,
    # )

    # agent.train(
    #     env_santa_rnn,
    #     episodes=num_epochs,
    #     check_point_dir=f"./check_points/santa_rnn_{node}_{seed}/",
    # )
