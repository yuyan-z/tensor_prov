import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


class ProvTree:
    def __init__(self):
        self.G = nx.DiGraph()
        self.root = 'root'
        self.node_map = {}  # {"id": obj}

        # Init the root node
        self.G.add_node(self.root)

    def add_child(self, parent_obj, child_obj, edge_obj=None):
        parent_id = id(parent_obj)
        if not self.G.has_node(parent_id):
            self.G.add_node(parent_id)
            self.node_map[parent_id] = parent_obj
            self.G.add_edge(self.root, parent_id)

        child_id = id(child_obj)
        self.node_map[child_id] = child_obj

        # Add a node by id
        if not self.G.has_node(child_id):
            self.G.add_node(child_id)

        # Add an edge by object
        if not self.G.has_edge(parent_id, child_id):
            self.G.add_edge(parent_id, child_id, obj=edge_obj)

        return child_id

    def get_all_paths(self):
        leaves = [node for node in self.G.nodes if self.G.out_degree(node) == 0 and node != self.root]
        all_paths = []
        for leaf in leaves:
            for nodes in nx.all_simple_paths(self.G, source=self.root, target=leaf):
                all_paths.append([self.node_map[node] for node in nodes[1:]])
        return all_paths

    def get_edges(self, start_obj, end_obj):
        start_id = id(start_obj)
        end_id = id(end_obj)
        if not self.G.has_node(start_id) or not self.G.has_node(end_id):
            return []

        edge_objs_list = []
        for path in nx.all_simple_paths(self.G, source=start_id, target=end_id):
            edges_objs = []
            for u, v in zip(path[:-1], path[1:]):
                edges_objs.append(self.G.edges[u, v].get('obj'))
            edge_objs_list.append(edges_objs)
        return edge_objs_list

    def visualize(self, figsize=(12, 6), show_values=True, seed=42):
        pos = nx.spring_layout(self.G, seed=seed)
        labels = {}
        for n in self.G.nodes:
            if n == self.root:
                labels[n] = 'root'
            else:
                obj = self.node_map[n]
                if show_values:
                    if isinstance(obj, pd.DataFrame):
                        labels[n] = getattr(obj, 'name', str(obj))
                    else:
                        labels[n] = str(obj)
                else:
                    labels[n] = str(n)

        plt.figure(figsize=figsize)
        nx.draw(self.G, pos, with_labels=True, labels=labels, node_size=1500, node_color='lightblue', font_size=10,
                arrowsize=20)
        plt.show()
