from typing import Tuple, Union

import numpy as np

from ..graph.vrp_network import VRPNetwork
from .common import ObsType
from .tsp import TSPEnv


class SantaIRPEnv(TSPEnv):
    """
    IRPEnv implements the Inventory Routing Problem a variant
    of the Vehicle Routing Problem. The vehicle has a
    capacity of 1. Visiting a node is only allowed if the
    cars capacity is greater or equal than the nodes demand.

    State: Shape (batch_size, num_nodes, 5) The third
        dimension is structured as follows:
        [x_coord, y_coord, demand, is_depot, visitable]

    Actions: Depends on the number of nodes in every graph.
        Should contain the node numbers to visit next for
        each graph. Shape (batch_size, 1)
    """

    # Update
    # Added these for rewards
    PRESENT_DELIVERY_REWARD = 10                # Reward for delivering a present
    COAL_DELIVERY_REWARD = 5                    # Reward for delivering coal
    ENERGY_DEPLETION_METHOD = "end_episode"     # Either 'end_episode' or 'return_to_depot'
    ENERGY_DEPLETION_PENALTY = 99               # Penalty for running out of energy
    ENERGY_COST_PER_DISTANCE = 1.0              # Base cost per unit distance: This may need some thought so long distances are not penalised as much???

    metadata = {"render.modes": ["human", "rgb_array"]}

    # Update
    # Added in initial_energy variable and comment
    def __init__(
        self,
        num_nodes: int = 32,
        batch_size: int = 128,
        num_draw: int = 6,
        seed: int = 69,
        initial_energy: float = 100.0,
    ):
        """
        Args:
            num_nodes (int, optional): Number of nodes in each generated graph. Defaults to 32.
            batch_size (int, optional): Number of graphs to generate. Defaults to 128.
            num_draw (int, optional): When calling the render num_draw graphs will be rendered. 
                Defaults to 6.
            seed (int, optional): Seed of the environment. Defaults to 69.
            initial_energy (float, optional): Initial energy for Santa's reindeer. Defaults to 100.0.
        """
        super().__init__(
            num_nodes=num_nodes, batch_size=batch_size, num_draw=num_draw, seed=seed,
        )

        # Update
        self.load = np.zeros(shape=(batch_size)) # 0 for nothing, 1 for present, 2 for coal
        # Differentiate Between Good and Bad Children:
        self.child_type = np.random.choice([0, 1], size=(self.batch_size, self.num_nodes)) # 0 for bad (coal), 1 for good (present) 
        # Implement Energy Constraints:
        self.initial_energy = initial_energy
        self.energy = np.full(self.batch_size, self.initial_energy)  # Define initial_energy



    # Update
    # Energy cost and Wind Effect:
    def calculate_energy_cost(self, traversed_edges):
        # Calculate the distance between each pair of nodes in traversed_edges
        # traversed_edges is an array of shape (batch_size, 2) where each row is [from_node, to_node]
        from_nodes = traversed_edges[:, 0]
        to_nodes = traversed_edges[:, 1]

        # Retrieve the coordinates of these nodes
        from_coords = self.sampler.get_graph_positions()[np.arange(self.batch_size), from_nodes]
        to_coords = self.sampler.get_graph_positions()[np.arange(self.batch_size), to_nodes]

        # Calculate Euclidean distance
        distances = np.sqrt(np.sum((from_coords - to_coords) ** 2, axis=1))

        # Calculate base energy cost
        energy_cost = distances * self.ENERGY_COST_PER_DISTANCE

        # Apply wind effect
        wind_effect = np.random.uniform(low=1.0, high=1.5, size=self.batch_size)
        return energy_cost * wind_effect



    # Update
    # Removed step function and replaced with a new one
    #def step(self, actions: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Run the environment one timestep. It's the users responsiblity to
        call reset() when the end of the episode has been reached. Accepts
        an actions and return a tuple of (observation, reward, done, info)

        Args:
            actions (nd.ndarray): Which node to visit for each graph.
                Shape of actions is (batch_size, 1).

        Returns:
            Tuple[ObsType, float, bool, dict]: Tuple of the observation,
                reward, done and info. The observation is within
                self.observation_space. The reward is for the previous action.
                If done equals True then the episode is over. Stepping through
                environment while done returns undefined results. Info contains
                may contain additions info in terms of metrics, state variables
                and such.
        """
        """
        assert (
            actions.shape[0] == self.batch_size
        ), "Number of actions need to equal the number of generated graphs."

        self.step_count += 1

        # visit each next node
        self.visited[np.arange(len(actions)), actions.T] = 1
        traversed_edges = np.hstack([self.current_location, actions]).astype(int)
        self.sampler.visit_edges(traversed_edges)

        # get demand of the visited nodes
        selected_demands = self.demands[
            np.arange(len(self.demands)), actions.T
        ].squeeze()

        # update load of each vehicle
        self.load -= selected_demands
        self.load[np.where(actions == self.depots)[0]] = 1

        self.current_location = np.array(actions)

        if self.video_save_path is not None:
            self.vid.capture_frame()

        done = self.is_done()
        return (
            self.get_state(),
            -self.sampler.get_distances(traversed_edges),
            done,
            None,
        )
        """
    # Update
    # Added new step function
    def step(self, actions: np.ndarray):
        # Existing code to update the environment's state based on actions
        # This includes moving to the next node, updating visited nodes, etc.
        super().step(actions)

        # Initialize reward array
        reward = np.zeros(self.batch_size)

        # Calculate the distance and energy cost
        traversed_edges = np.hstack([self.current_location, actions]).astype(int)
        energy_cost = self.calculate_energy_cost(traversed_edges)
        self.energy -= energy_cost

        # Update state and deliver presents/coal
        for i in range(self.batch_size):
            # Update logic for delivering presents and coal
            if self.child_type[i, actions[i]] == 1 and self.load[i] == 1:
                self.load[i] = 0  # Present delivered
                reward[i] += self.PRESENT_DELIVERY_REWARD
            elif self.child_type[i, actions[i]] == 0 and self.load[i] == 2:
                self.load[i] = 0  # Coal delivered
                reward[i] += self.COAL_DELIVERY_REWARD

            # Penalise for energy inefficiency
            reward[i] -= energy_cost[i]  # Scale this???

        # Check for energy depletion and handle
        for i in range(self.batch_size):
            if self.energy[i] <= 0:
                if self.ENERGY_DEPLETION_METHOD == "end_episode":
                    self.done[i] = True
                    reward[i] -= self.ENERGY_DEPLETION_PENALTY
                else:  # "return_to_depot"
                    self.current_location[i] = self.depots[i]
                    self.energy[i] = self.initial_energy
                    reward[i] -= self.ENERGY_DEPLETION_PENALTY

        # Determine if the episode is done
        done = self.is_done()

        return (self.get_state(), reward, done, None)


    # Update
    # Removed this get_state function and replaced with another
    #def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Getter for the current environment state.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Shape (num_graph, num_nodes, 5)
                The third dimension is structured as follows:
            [x_coord, y_coord, demand, is_depot, visitable]
        """
        """
        # generate state (depots not yet set)
        state = np.dstack(
            [
                self.sampler.get_graph_positions(),
                self.demands,
                np.zeros((self.batch_size, self.num_nodes)),
                self.generate_mask(),
            ]
        )

        # set depots in state to 1
        state[np.arange(len(state)), self.depots.T, 3] = 1

        return (state, self.load)
        """
    # Update
    # Added new get_state function with extra variables
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the current environment state split into graph state and vehicle state.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
            - Graph state with shape (batch_size, num_nodes, 5)
            - Vehicle state with shape (batch_size, 2)
        """
        
        full_state = super().get_state()

        # Split the state into graph state and vehicle state
        graph_state = full_state[:, :, :5]  # First 5 elements for graph state
        vehicle_state = np.zeros((self.batch_size, 2))  # Initialize vehicle state array

        # Populate vehicle state with energy and load
        vehicle_state[:, 0] = self.energy  # Energy
        vehicle_state[:, 1] = self.load    # Load

        return graph_state, vehicle_state

    def generate_mask(self):
        """
        Generates a mask of where the nodes marked as 1 cannot
        be visited in the next step according to the env dynamic.

        Returns:
            np.ndarray: Returns mask for each (un)visitable node
                in each graph. Shape (batch_size, num_nodes)
        """

        # disallow staying at the depot
        depot_graphs_idxs = np.where(self.current_location == self.depots)[0]
        self.visited[depot_graphs_idxs, self.depots[depot_graphs_idxs].squeeze()] = 1

        # allow visiting the depot when not currently at the depot
        depot_graphs_idxs_not = np.where(self.current_location != self.depots)[0]
        self.visited[
            depot_graphs_idxs_not, self.depots[depot_graphs_idxs_not].squeeze()
        ] = 0

        # allow staying on a depot if the graph is solved.
        done_graphs = np.where(np.all(self.visited, axis=1) == True)[0]
        self.visited[done_graphs, self.depots[done_graphs].squeeze()] = 0

        # disallow visiting nodes that exceed the current load.
        mask = np.copy(self.visited)
        exceed_demand_idxs = ((self.demands - self.load[:, None, None]) > 0).squeeze()
        mask[exceed_demand_idxs] = 1

        return mask

    def generate_graphs(self):
        """
        Generates a VRPNetwork of batch_size graphs with num_nodes
        each. Resets the visited nodes to 0.
        """
        self.visited = np.zeros(shape=(self.batch_size, self.num_nodes))
        self.sampler = VRPNetwork(
            num_graphs=self.batch_size,
            num_nodes=self.num_nodes,
            num_depots=1,
            plot_demand=True,
        )

        # set current location to the depots
        self.depots = self.sampler.get_depots()
        self.current_location = self.depots

        self.demands = self.sampler.get_demands()

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        """
        Resets the environment.

        Returns:
            Union[ObsType, Tuple[ObsType, dict]]: State of the environment.
        """
        super().reset()
        self.load = np.ones(shape=(self.batch_size,))
        return self.get_state()
