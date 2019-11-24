import matplotlib.pyplot as plt
import networkx as nx
import random

from distributions import constant, uniform, zipf
from factories import ClusteredGraphBuilder
from collections import Counter


def optimize_choice(states: list):
    most_common = Counter(states).most_common(1)[0][0]

    if most_common.lower() == "rock":
        return "paper"
    elif most_common.lower() == "paper":
        return "scissors"
    elif most_common.lower() == "scissors":
        return "rock"
    else:
        raise Exception("invalid state!")


if __name__ == "__main__":
    graph = (
        ClusteredGraphBuilder()
        .set_graph_size(nodes=40)
        .with_edge_degree_distribution(zipf, a=2)
        .with_triangle_degree_distribution(constant, k=3)
        .initialize_node_attributes_using({'states': list, 'next': str})
        .build()
    )

    for node, data in graph.nodes(data=True):
        data['states'].append(random.choice(("rock", "paper", "scissors")))

    for _ in range(50):
        for node, data in graph.nodes(data=True):
            neigh = graph[node]
            states = [graph.nodes[n]['states'][-1] for n in neigh]
            data["next"] = optimize_choice(states)

        for node, data in graph.nodes(data=True):
            data['states'].append(data['next'])
            data['next'] = ""

    nx.draw_kamada_kawai(graph)
    plt.show()


