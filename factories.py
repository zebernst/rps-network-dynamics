import random
from functools import partial
from itertools import combinations, repeat
from typing import Callable, Optional

import networkx as nx
from distributions import zipf, uniform, constant


class ClusteredGraphBuilder:
    def __init__(self):
        self.num_nodes: Optional[int] = None
        self.get_edge_degree: Optional[Callable[[], int]] = None
        self.get_triangle_degree: Optional[Callable[[], int]] = None
        self.attribute_factory: Optional[dict] = None

    def set_graph_size(self, nodes: int) -> "ClusteredGraphBuilder":
        self.num_nodes = nodes
        return self

    def with_edge_degree_distribution(
        self, distribution: Callable[..., int], **kwargs: int
    ) -> "ClusteredGraphBuilder":
        if distribution == zipf:
            if "a" not in kwargs:
                raise nx.NetworkXError(
                    "zipf distribution requires an 'a' parameter to be set."
                )
        elif distribution == uniform:
            if "high" not in kwargs or "low" not in kwargs:
                raise nx.NetworkXError(
                    "uniform distribution requires both a 'high' "
                    "and a 'low' parameter to be set."
                )
        elif distribution == constant:
            if "k" not in kwargs:
                raise nx.NetworkXError(
                    "constant distribution requires a 'k' parameter to be set."
                )
            else:
                if (kwargs["k"] * self.num_nodes) % 2 != 0:
                    raise nx.NetworkXError(
                        "invalid constant distribution for edge degree. the product of the k parameter "
                        "and the number of nodes must be divisible by two."
                    )
        else:
            raise nx.NetworkXError("invalid distribution.")

        self.get_edge_degree = partial(distribution, **kwargs)
        return self

    def with_triangle_degree_distribution(
        self, distribution: Callable[..., int], **kwargs: int
    ) -> "ClusteredGraphBuilder":
        if distribution == zipf:
            if "a" not in kwargs:
                raise nx.NetworkXError(
                    "zipf distribution requires an 'a' parameter to be set."
                )
        elif distribution == uniform:
            if "high" not in kwargs or "low" not in kwargs:
                raise nx.NetworkXError(
                    "uniform distribution requires both a 'high' "
                    "and a 'low' parameter to be set."
                )
        elif distribution == constant:
            if "k" not in kwargs:
                raise nx.NetworkXError(
                    "constant distribution requires a 'k' parameter to be set."
                )
            else:
                if (kwargs["k"] * self.num_nodes) % 2 != 0:
                    raise nx.NetworkXError(
                        "invalid constant distribution for triangle degree. the product of the k parameter "
                        "and the number of nodes must be divisible by three."
                    )
        else:
            raise nx.NetworkXError("invalid distribution.")

        self.get_triangle_degree = partial(distribution, **kwargs)
        return self

    def initialize_node_attributes_using(self, data: dict) -> "ClusteredGraphBuilder":
        self.attribute_factory = data
        return self

    def build(self) -> nx.Graph:
        if any(
            attr is None
            for attr in (
                self.num_nodes,
                self.get_edge_degree,
                self.get_triangle_degree,
            )
        ):
            raise nx.NetworkXError("graph builder not properly configured.")

        G = nx.empty_graph(self.num_nodes)

        while True:
            edges = [i for n in G for i in repeat(n, self.get_edge_degree())]
            triangles = [i for n in G for i in repeat(n, self.get_triangle_degree())]

            random.shuffle(edges)
            random.shuffle(triangles)

            if len(edges) % 2 == 0 and len(triangles) % 3 == 0:
                break

        while edges:
            G.add_edge(edges.pop(), edges.pop())
        while triangles:
            G.add_edges_from(combinations((triangles.pop() for _ in range(3)), 2))

        if self.attribute_factory is not None:
            for node, data in G.nodes(data=True):
                for k, factory in self.attribute_factory.items():
                    data[k] = factory()

        return G
