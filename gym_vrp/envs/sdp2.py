import numpy as np
import torch
from .irp import IRPEnv

class SantaIRPEnv(IRPEnv):
    
    """
    SantaIRPEnv implements the Santa Inventory Routing Problem a variant
    of the Vehicle Routing Problem. The vehicle has a
    capacity of 1. Visiting a node is only allowed if the
    cars capacity is greater or equal than the nodes demand.

    State: Shape (batch_size, num_nodes, 6) The third
        dimension is structured as follows:
        [x_coord, y_coord, demand, is_depot, visitable, good_or_bad]

    Actions: Depends on the number of nodes in every graph.
        Should contain the node numbers to visit next for
        each graph. Shape (batch_size, 1)
    """
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = np.random.seed(23)

        self.child_behavior = np.random.choice(['good', 'bad'], size=self.num_nodes)
        
        # Configurable reward and penalty values
        # self.max_energy = self.num_nodes * 100
        self.max_energy = 75
        self.energy = self.max_energy
        self.energy_depletion_penalty = -50

        # Wind factor related variables
        self.energy_strategy = "return" # "stop": Stops the run or "return": Back to depot, apply penalty and continue (default)
        # self.base_energy_consumption_rate = self.num_nodes
        self.base_energy_consumption_rate = 10
        self.wind_factor_range = (0.5, 1.5)  # Wind can decrease or increase energy consumption

        # Santa can carry multiple items
        self.santa_carrying = {'present': 0, 'coal': 0}
        self.pickup()
        
    def reset(self):
        state = super().reset()
        self.child_behavior = np.random.choice([1, 0], size=self.num_nodes)
        self.energy = self.max_energy
        self.santa_carrying = {'present': 0, 'coal': 0}
        self.pickup()
        return state
    
    def pickup(self):
        item_choice = np.random.choice(['present', 'coal'])
        self.santa_carrying = {'present': 1, 'coal': 0} if item_choice == "present" else {'present': 0, 'coal': 1}

    def step(self, action):

        observation, reward, done, info = super().step(action)
        current_node = action[0]

        # Determine the wind factor for this step
        wind_factor = np.random.uniform(*self.wind_factor_range)

        # Adjust energy consumption based on wind factor
        energy_consumption = self.base_energy_consumption_rate * wind_factor
        self.energy -= energy_consumption

        #print(f"energy = {self.energy} and energy_consumption = {energy_consumption}")
        #print(f"{self.energy}, ", end="")
        if self.energy <= 0:
            if self.energy_strategy == "stop":
                # This just stops it for now, we may consider other ways to deal with this
                # Apply energy depletion penalty
                reward -= self.energy_depletion_penalty
                done = True
            else: # "return"
                # This returns to the depot
                # Apply energy depletion penalty
                reward -= self.energy_depletion_penalty

                # Automatically return Santa to the depot
                current_node = self.depots[0]

                # print("Run out of energy, penalty applied")
                print("!", end="")

        # Check if Santa is at the depot
        if current_node == self.depots[0]:
            # Replenish energy and items at the depot
            self.energy = self.max_energy
            # Present/Coal pick up strategy
            self.pickup()

        return observation, reward, done, info
    
    def get_state(self):
        state, load = super().get_state()

        batch_size, num_nodes, _ = state.shape

        # Ensure child_behavior is numerical and correctly shaped
        if self.child_behavior.ndim == 1:
            child_behavior_expanded = np.tile(self.child_behavior, (batch_size, 1))
        else:
            child_behavior_expanded = self.child_behavior

        child_behavior_state = child_behavior_expanded.reshape(batch_size, num_nodes, 1)
        updated_state = np.concatenate((state, child_behavior_state), axis=-1)

        return updated_state, load

    
    def generate_mask(self):
        mask = super().generate_mask()

        # Get indices of good and bad children
        good_child_indices = np.where(self.child_behavior == 'good')[0]
        bad_child_indices = np.where(self.child_behavior == 'bad')[0]

        if self.santa_carrying['present'] == 0:
            # If Santa has no presents, mark nodes with good children as unvisitable
            mask[good_child_indices] = 1
        if self.santa_carrying['coal'] == 0:
            # If Santa has no coal, mark nodes with bad children as unvisitable
            mask[bad_child_indices] = 1

        return mask