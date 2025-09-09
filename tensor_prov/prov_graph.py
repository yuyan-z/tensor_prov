import json
import os

import networkx as nx
import itertools

import numpy as np
from matplotlib import pyplot as plt


class ProvGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self._id_counter = itertools.count(1)
        self.root = 'root'
        self.G.add_node(self.root)

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

    def save_graph(self, file_dir: str) -> dict:
        graph_json = {
            "root": self.root,
            "nodes": [n for n in self.G.nodes() if n != self.root],
            "edges": [{"src": u, "dst": v} for u, v in self.G.edges()]
        }
        file_path = os.path.join(file_dir, "graph.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, ensure_ascii=False, indent=2)
        return graph_json

    def load_graph(self, file_dir: str):
        file_path = os.path.join(file_dir, "graph.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, "r", encoding="utf-8") as f:
            graph_json = json.load(f)

        self.G = nx.DiGraph()
        self.root = graph_json.get("root", "root")
        nodes = graph_json.get("nodes", [])

        # Add nodes
        for n in nodes:
            self.G.add_node(n)

        # Add edges
        for e in graph_json.get("edges", []):
            u, v = e["src"], e["dst"]
            self.G.add_edge(u, v)

        # Set _id_counter
        next_id = (max(nodes) + 1) if nodes else 1
        self._id_counter = itertools.count(next_id)
