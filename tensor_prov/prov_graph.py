import networkx as nx
import itertools

from matplotlib import pyplot as plt


class ProvGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self._id_counter = itertools.count(1)
        self.root = 'root'

    def new_id(self):
        return next(self._id_counter)

    def add_child(self, parent_id, child_id, edge_obj=None):
        # Add the parent node
        if not self.G.has_node(parent_id):
            self.G.add_node(parent_id)
            self.G.add_edge(self.root, parent_id)

        # Add the child node
        if not self.G.has_node(child_id):
            self.G.add_node(child_id)

        # Add the edge
        if not self.G.has_edge(parent_id, child_id):
            self.G.add_edge(parent_id, child_id, obj=edge_obj)

    def get_edges(self, start_id, end_id):
        if not self.G.has_node(start_id) or not self.G.has_node(end_id):
            return []

        edge_objs_list = []
        for path in nx.all_simple_paths(self.G, source=start_id, target=end_id):
            edges_objs = []
            for u, v in zip(path[:-1], path[1:]):
                edges_objs.append(self.G.edges[u, v].get('obj'))
            edge_objs_list.append(edges_objs)
        return edge_objs_list

    def visualize(self, figsize=(12, 6)):
        pos = nx.spring_layout(self.G, seed=42)
        # labels = {}
        # for n in self.G.nodes:
        #     if n == self.root:
        #         labels[n] = 'root'
        #     else:
        #         labels[n] =

        plt.figure(figsize=figsize)
        nx.draw(self.G, pos, node_size=1500, node_color='lightblue', font_size=10,
                arrowsize=20)
        plt.show()
