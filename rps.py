import matplotlib.pyplot as plt
import networkx as nx

from distributions import constant, uniform, zipf
from factories import ClusteredGraphBuilder


if __name__ == "__main__":
    graph = (
        ClusteredGraphBuilder()
        .set_graph_size(nodes=40)
        .with_edge_degree_distribution(zipf, a=2)
        .with_triangle_degree_distribution(constant, k=3)
        .initialize_node_attributes_using({'states': list, 'next': str})
        .build()
    )

    nx.draw_kamada_kawai(graph)
    plt.show()
