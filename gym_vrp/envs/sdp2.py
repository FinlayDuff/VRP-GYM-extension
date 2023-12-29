import numpy as np
from .irp import IRPEnv

class SantaIRPEnv(IRPEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.child_behavior = None
        self.energy = None
        self.max_energy = 100

        self.seed = np.random.seed(23)

        # Configurable reward and penalty values
        self.correct_delivery_reward = 10
        self.incorrect_delivery_penalty = -10
        self.energy_depletion_penalty = -50

        # Wind factor related variables
        self.energy_strategy = "stop" # "stop": Stops the run or "return": Back to depot, apply penalty and continue (default)
        self.base_energy_consumption_rate = 4
        self.wind_factor_range = (0.8, 1.2)  # Wind can decrease or increase energy consumption

        # Santa can carry multiple items
        self.santa_carrying = {'present': 0, 'coal': 0}
        self.max_presents = 2
        self.max_coal = 2
        self.pickup_stragey = "either" # Either: Pickus up either 1 coal or 1 present (defualt), "both": Picks up both upto max, "random": picks up a random between 0 and max for each
        
    def reset(self):
        state = super().reset()
        self.child_behavior = np.random.choice(['good', 'bad'], size=self.num_nodes)
        self.energy = self.max_energy
        self.santa_carrying = {'present': 0, 'coal': 0}
        return state

    def step(self, action):
        observation, reward, done, info = super().step(action)
        current_node = action[0]

        # Determine the wind factor for this step
        wind_factor = np.random.uniform(*self.wind_factor_range)

        # Adjust energy consumption based on wind factor
        energy_consumption = self.base_energy_consumption_rate * wind_factor
        self.energy -= energy_consumption

        if self.energy <= 0:
            if self.energy_strategy == "stop":
                # This just stops it for now, we may consider other ways to deal with this
                # Apply energy depletion penalty
                reward += self.energy_depletion_penalty
                done = True
            else:
                # Apply energy depletion penalty
                reward += self.energy_depletion_penalty

                # Automatically return Santa to the depot
                current_node = self.depots[0]
                self.energy = self.max_energy

                # Replenish items at the depot based on the pickup strategy
                if self.pickup_stragey == 'both':
                    self.santa_carrying = {'present': self.max_presents, 'coal': self.max_coal}
                elif self.pickup_stragey == 'random':
                    self.santa_carrying = {'present': np.random.randint(0, self.max_presents + 1), 'coal': np.random.randint(0, self.max_coal + 1)}
                else:  # 'either'
                    item_choice = np.random.choice(['present', 'coal'])
                    self.santa_carrying = {'present': 1, 'coal': 0} if item_choice == "present" else {'present': 0, 'coal': 1}

        # Check if the delivery is correct
        if self.child_behavior[current_node] == 'good' and self.santa_carrying == 'present':
            reward += self.correct_delivery_reward
        elif self.child_behavior[current_node] == 'bad' and self.santa_carrying == 'coal':
            reward += self.correct_delivery_reward
        else:
            reward += self.incorrect_delivery_penalty

        # Update what Santa is carrying
        self.santa_carrying = 'present' if self.santa_carrying == 'coal' else 'coal'

        # Check if Santa is at the depot
        if current_node == self.depots[0]:
            # Replenish energy and items at the depot
            self.energy = self.max_energy
            # Present/Coal pick up strategy
            if self.pickup_stragey == 'both':
                self.santa_carrying = {'present': self.max_presents, 'coal': self.max_coal}
            elif self.pickup_stragey == 'random':
                self.santa_carrying = {'present': np.random.randint(0, self.max_presents + 1), 'coal': np.random.randint(0, self.max_coal + 1)}
            else:  # 'either'
                item_choice = np.random.choice(['present', 'coal'])
                self.santa_carrying = {'present': 1, 'coal': 0} if item_choice == "present" else {'present': 0, 'coal': 1}
        else:
            # Deliver items and update santa_carrying
            if self.child_behavior[current_node] == 'good' and self.santa_carrying['present'] > 0:
                reward += self.correct_delivery_reward
                self.santa_carrying['present'] -= 1
            elif self.child_behavior[current_node] == 'bad' and self.santa_carrying['coal'] > 0:
                reward += self.correct_delivery_reward
                self.santa_carrying['coal'] -= 1
            else:
                reward += self.incorrect_delivery_penalty

        return observation, reward, done, info
