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

def conv_cycle(states: list):
    length=2
    converged=False
    while not converged and length<20:
        converged=True
        for i in range(1,length+1):
            matchstate=states[-i]
            if states[-(i+length)] != matchstate:
                converged=False
        length +=1
    if converged:
        return length-1
    else:
        raise Exception("Network did not Converge! Run for more itterations")

if __name__ == "__main__":
    graph = (
        ClusteredGraphBuilder()
        .set_graph_size(nodes=100)
        .with_edge_degree_distribution(uniform, low=1, high=6)
        .with_triangle_degree_distribution(uniform, low=0, high=3)
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

    conv_cycles=[]
    for node, data in graph.nodes(data=True):
        conv_cycles.append(conv_cycle(data['states']))
    print(conv_cycles)

