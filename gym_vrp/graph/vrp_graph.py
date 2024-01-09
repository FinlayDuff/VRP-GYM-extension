import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

class VRPGraph:

    graph: nx.Graph = nx.Graph()

    def __init__(self, num_nodes: int, num_depots: int, plot_demand: bool = False, this_energy=None, this_load=None):
        """
        Creates a fully connected graph with node_num nodes
        and depot num depots. Coordinates of each node
        and the depot nodes will be samples randomly.

        Args:
            node_num (int): Number of nodes in the graph.
            depot_num (int): Number of depots in the graph.
        """

        self.this_energy = this_energy
        self.this_load = this_load
        
        self.num_nodes = num_nodes
        self.num_depots = num_depots
        self.plot_demand = plot_demand

        # offset for node labels
        self.offset = offset = np.array([0, 0.065])

        # generate graph and set node position
        self.graph = nx.complete_graph(num_nodes)
        node_position = {
            i: coordinates for i, coordinates in enumerate(np.random.rand(num_nodes, 2))
        }
        nx.set_node_attributes(self.graph, node_position, "coordinates")

        # sample depots and set attributes
        self.depots = np.random.choice(num_nodes, size=num_depots, replace=False)
        one_hot = np.zeros(num_nodes)
        one_hot[self.depots] = 1
        one_hot_dict = {i: depot for i, depot in enumerate(one_hot)}
        nx.set_node_attributes(self.graph, one_hot_dict, "depot")

        # set demand for each node except depots
        C = 0.2449 * num_nodes + 26.12  # linear reg on values from paper
        demand = np.random.uniform(low=1, high=10, size=(num_nodes, 1)) / C
        demand[self.depots] = 0
        node_demand = {i: d for i, d in enumerate(demand)}
        nx.set_node_attributes(self.graph, node_demand, "demand")

        self.set_default_node_attributes()

    def set_default_node_attributes(self):
        """
        Sets the default colors of the  nodes
        as attributes. Nodes are black except
        depots which are colored in red.

        Edges are initially marked as unvisited.
        """
        nx.set_edge_attributes(self.graph, False, "visited")
        nx.set_node_attributes(self.graph, "black", "node_color")
        for node in self.depots:
            self.graph.nodes[node]["node_color"] = "red"

    def draw(self, ax):
        """
        Draws the graph as a matplotlib plot.
        Depots are colored in red. Edges that have been
        traveresed 
        """

        # draw nodes according to color and position attribute
        pos = nx.get_node_attributes(self.graph, "coordinates")
        node_colors = nx.get_node_attributes(self.graph, "node_color").values()
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_colors, ax=ax, node_size=100
        )

        # draw edges that where visited
        # edges = [x for x in self.graph.edges(data=True) if x[2]["visited"]]
        # nx.draw_networkx_edges(
        #     self.graph,
        #     pos,
        #     alpha=0.5,
        #     edgelist=edges,
        #     edge_color="red",
        #     ax=ax,
        #     width=1.5,
        # )

        # draw visited edges
        visited_edges = [edge for edge in self.graph.edges(data=True) if edge[2]['visited']]
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=visited_edges,
            edge_color='red',
            ax=ax,
            width=1.5
        )
        
        # draw edges that lead back to the depot with a dashed blue line
        depot_edges = [edge for edge in visited_edges if edge[0] in self.depots or edge[1] in self.depots]
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=depot_edges,
            style='dashed',
            edge_color='blue',
            ax=ax,
            width=1.5
        )


        # draw demand above the node

        if self.plot_demand:



            demand_label_pos = {k: (v + self.offset) for k, v in pos.items()}
            node_demand = nx.get_node_attributes(self.graph, "demand")
            node_demand = {k: np.round(v, 2)[0] for k, v in node_demand.items()}

            # node_demand = {key: 'D:{:.2f}'.format(value) for key, value in node_demand.items()}

            nx.draw_networkx_labels(
                self.graph, demand_label_pos, labels=node_demand, ax=ax
            )

        # Format the energy and load values to two decimal places
        energy_text = f'Energy: {self.this_energy:.2f}'
        load_text = f'Load: {self.this_load:.2f}'

        # Plot the formatted energy and load in the top left corner
        plt.text(0.05, 0.95, energy_text, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', fontsize=10, color='blue')
        plt.text(0.05, 0.90, load_text, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', fontsize=10, color='green')




    def visit_edge(self, source_node: int, target_node: int) -> None:
        """
        Sets the edge color to red.

        Args:
            source_node (int): Source node id of the edge
            target_node (int): Target node id of the edge
        """

        # don't draw self loops
        if source_node == target_node:
            return

        self.graph.edges[source_node, target_node]["visited"] = True

    @property
    def demand(self) -> np.ndarray:
        positions = nx.get_node_attributes(self.graph, "demand").values()
        return np.asarray(list(positions))

    @property
    def edges(self):
        return self.graph.edges.data()

    @property
    def nodes(self):
        return self.graph.nodes.data()

    @property
    def node_positions(self) -> np.ndarray:
        """
        Returns the coordinates of each node as
        an ndarray of shape (num_nodes, 2) sorted
        by the node index.
        """

        positions = nx.get_node_attributes(self.graph, "coordinates").values()
        return np.asarray(list(positions))

    def euclid_distance(self, node1_idx: int, node2_idx: int) -> float:
        """
        Calculates the euclid distance between two nodes
        with their idx's respectively.
        """

        node_one_pos = self.graph.nodes[node1_idx]["coordinates"]
        node_two_pos = self.graph.nodes[node2_idx]["coordinates"]

        return np.linalg.norm(node_one_pos - node_two_pos)
