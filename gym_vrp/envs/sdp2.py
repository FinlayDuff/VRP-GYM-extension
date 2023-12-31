import numpy as np
from .irp import IRPEnv

class SantaIRPEnv(IRPEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = np.random.seed(23)
        self.child_behavior = None
        
        # Configurable reward and penalty values
        self.max_energy = 100
        self.energy = self.max_energy
        self.correct_delivery_reward = 10
        self.incorrect_delivery_penalty = -10
        self.energy_depletion_penalty = -50

        # Wind factor related variables
        self.energy_strategy = "return" # "stop": Stops the run or "return": Back to depot, apply penalty and continue (default)
        self.base_energy_consumption_rate = 4
        self.wind_factor_range = (0.8, 1.2)  # Wind can decrease or increase energy consumption

        # Santa can carry multiple items
        self.pickup_stragey = "either" # Either: Pickus up either 1 coal or 1 present (defualt), "both": Picks up both upto max, "random": picks up a random between 0 and max for each
        self.santa_carrying = {'present': 0, 'coal': 0}
        self.pickup()
        self.max_presents = 1
        self.max_coal = 1
        
    def reset(self):
        state = super().reset()
        self.child_behavior = np.random.choice(['good', 'bad'], size=self.num_nodes)
        self.energy = self.max_energy
        self.santa_carrying = {'present': 0, 'coal': 0}
        self.pickup()
        return state
    
    def pickup(self):
        if self.pickup_stragey == 'both':
            self.santa_carrying = {'present': self.max_presents, 'coal': self.max_coal}
        elif self.pickup_stragey == 'random':
            self.santa_carrying = {'present': np.random.randint(0, self.max_presents + 1), 'coal': np.random.randint(0, self.max_coal + 1)}
        else:  # 'either'
            item_choice = np.random.choice(['present', 'coal'])
            self.santa_carrying = {'present': 1, 'coal': 0} if item_choice == "present" else {'present': 0, 'coal': 1}

    def step(self, action):
        #print("Step method called. self.santa_carrying:", self.santa_carrying)

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

        # Check if Santa is at the depot
        if current_node == self.depots[0]:
            # Replenish energy and items at the depot
            self.energy = self.max_energy
            # Present/Coal pick up strategy
            self.pickup()
        else:
            # Deliver items and update santa_carrying
            # Check if the delivery is correct
            #print("Before accessing self.santa_carrying:", self.santa_carrying)
            if self.child_behavior[current_node] == 'good':
                if self.santa_carrying['present'] > 0:
                    # Correct delivery of present
                    reward += self.correct_delivery_reward
                    self.santa_carrying['present'] -= 1  # Update the number of presents Santa is carrying
                else:
                    # Incorrect delivery (Santa doesn't have a present for a good child)
                    reward += self.incorrect_delivery_penalty

            elif self.child_behavior[current_node] == 'bad':
                if self.santa_carrying['coal'] > 0:
                    # Correct delivery of coal
                    reward += self.correct_delivery_reward
                    self.santa_carrying['coal'] -= 1  # Update the number of coal Santa is carrying
                else:
                    # Incorrect delivery (Santa doesn't have coal for a bad child)
                    reward += self.incorrect_delivery_penalty

        return observation, reward, done, info
