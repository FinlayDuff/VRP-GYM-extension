import numpy as np
import torch
from .irp import IRPEnv

from ..graph.vrp_network import VRPNetwork


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

        self.energy = None
        self.load = None
        super().__init__(*args, **kwargs)

        # Configurable reward and penalty values
        self.max_energy = 1
        self.energy = self.max_energy * np.ones(self.batch_size)
        self.energy_depletion_penalty = round(0.5 * self.num_nodes)

        super().__init__(*args, **kwargs)

        # Wind factor related variables
        self.energy_strategy = "return"  # "stop": Stops the run or "return": Back to depot, apply penalty and continue (default)
        self.base_energy_consumption_rate = self.max_energy * 0.2

        # Stochastic element: wind impacts the energy consumed by the vehicle each journey
        self.wind_factor_range = (
            0.5,
            1.5,
        )

    def reset(self):
        """
        Reset the energy of santa at each reset. Load is handled by the IRP reset
        """
        state = super().reset()

        self.energy = self.max_energy * np.ones(self.batch_size)

        return state

    def step(self, action):
        # super-init the step from the IRP environment
        observation, reward, done, info = super().step(action)

        # Determine the wind factor for this step
        wind_factor = np.random.uniform(
            self.wind_factor_range[0], self.wind_factor_range[1], self.batch_size
        )

        # Calculate energy consumption for each batch item based on the wind factor
        energy_consumption = self.base_energy_consumption_rate * wind_factor

        # Update self.energy for each batch item based on energy consumption
        self.energy -= energy_consumption

        # replenish energy if now at the depot
        mask = np.squeeze(self.current_location) == np.squeeze(self.depots)
        self.energy = np.where(mask, self.max_energy, self.energy)

        # This creates a boolean tensor indicating which elements have depleted energy
        depleted = self.energy <= 0

        # Apply energy depletion penalty
        reward -= self.energy_depletion_penalty * depleted.astype(float)

        # Handle different strategies
        if self.energy_strategy == "stop":
            # Set 'done' flag for instances with depleted energy
            done = depleted
        else:  # "return"
            # Here you handle the return strategy for depleted energy instances
            # Assuming current_node is a tensor representing the current node for each batch item
            # and self.depots is accessible as a tensor or list

            depleted_reshaped = depleted[:, np.newaxis]

            # Use np.where to update self.current_location for instances with depleted energy
            self.current_location = np.where(
                depleted_reshaped, self.depots, self.current_location
            )

            # Replenish energy
            mask = np.squeeze(self.current_location) == np.squeeze(self.depots)
            self.energy = np.where(mask, self.max_energy, self.energy)

            # Replenish load
            self.load = np.where(mask, 1, self.load)

        return observation, reward, done, info

    def get_state(self):
        """
        Getter for the current environment state.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Shape (num_graph, num_nodes, 7)
                The third dimension is structured as follows:
            [x_coord, y_coord, demand, is_depot, visitable, vehicle_load, vehicle_energy]
        """
        state, load = super().get_state()

        # Add on the load to the state so that each node has the current load of the vehicle
        load_reshaped = load[:, np.newaxis, np.newaxis]
        # Add on the energy to the state so that each node has the current energy of the vehicle
        energy_reshaped = self.energy[:, np.newaxis, np.newaxis]

        # Expand load_reshaped and energy_reshaped to match the shape of state
        load_expanded = np.tile(load_reshaped, (1, state.shape[1], 1))
        energy_expanded = np.tile(energy_reshaped, (1, state.shape[1], 1))

        # Concatenate the expanded arrays with the existing state
        updated_state = np.concatenate((state, load_expanded, energy_expanded), axis=2)
        return updated_state, load

    def generate_graphs(self):
        """
        Generates a VRPNetwork of batch_size graphs with num_nodes
        each. Resets the visited nodes to 0.
        """

        if not hasattr(self, 'energy'):
            self.energy = np.zeros(shape=(self.batch_size,))
        if self.energy is None:
            self.energy = np.zeros(shape=(self.batch_size,))

        if not hasattr(self, 'load'):
            self.load = np.zeros(shape=(self.batch_size,))
        if self.load is None:
            self.load = np.zeros(shape=(self.batch_size,))

        print(self.energy)

        self.visited = np.zeros(shape=(self.batch_size, self.num_nodes))
        self.sampler = VRPNetwork(
            num_graphs=self.batch_size,
            num_nodes=self.num_nodes,
            num_depots=1,
            plot_demand=True,
            energy=self.energy,
            load=self.load,
        )

        # set current location to the depots
        self.depots = self.sampler.get_depots()
        self.current_location = self.depots

        self.demands = self.sampler.get_demands()