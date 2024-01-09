<!-- ABOUT THE PROJECT -->
## About The Project
This is an extension to the VRP gym created here: https://github.com/kevin-schumann/VRP-GYM

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
To install the required packages we recommend using poetry
```sh
   poetry install
```

<!-- USAGE EXAMPLES -->
## Usage
To initialize an sdp environment and train an agent you just need a few lines of code. Suppose you want to train the FF agent on the SDP problem:
``` py
batch_size = 128
seed = 123
num_nodes = 7

num_epochs = 2001
lr=1E-3
gamma = 0.99
dropout_rate = 0.5
hidden_dim=1024


# Instantiate the SantaIRPEnv environment
env_santa = SantaIRPEnv(num_nodes=num_nodes, batch_size=batch_size, seed=seed, num_draw=3)

# Instantiate the TSPAgentFF (assuming it's compatible with SantaIRPEnv)
agent_santa_ff = SDPAgentFF(node_dim=num_nodes,hidden_dim=hidden_dim,lr=lr,gamma=gamma,dropout_rate=dropout_rate,
    seed=seed, csv_path=f"./train_logs/loss_log_santa_ff_{num_nodes}_{seed}.csv",
)


agent_santa_ff.train(
    env_santa,
    episodes=num_epochs,
    check_point_dir=f"./check_points/santa_ff_{num_nodes}_{seed}/",
)
```

The agent will save his progress every 50 epochs in a directory called `check_points`.

## Reproduction
To reproduce our results just run the `reproduction.ipynb` notebook. This will load the agents and plot the learning curve as well as track the mean reward by node.







