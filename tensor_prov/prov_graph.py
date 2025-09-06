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

    def add_edge(self, src_id, dst_id, edge_obj=None):
        # Add the source node
        if not self.G.has_node(src_id):
            self.G.add_node(src_id)
            self.G.add_edge(self.root, src_id)

        # Add the destination node
        if not self.G.has_node(dst_id):
            self.G.add_node(dst_id)

        # Add the edge
        if not self.G.has_edge(src_id, dst_id):
            self.G.add_edge(src_id, dst_id, obj=edge_obj)

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

    def visualize(self, figsize=(12, 6), seed=42):
        pos = nx.spring_layout(self.G, seed=seed)
        labels = {}
        for n in self.G.nodes:
            if n == self.root:
                labels[n] = 'root'
            else:
                labels[n] = n
        plt.figure(figsize=figsize)
        nx.draw(self.G, pos, with_labels=True, labels=labels, node_size=1500, node_color='lightblue', font_size=10,
                arrowsize=20)
        plt.show()
