import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
import random

from distributions import constant, uniform, zipf
from factories import *
from collections import Counter
from matplotlib.animation import FuncAnimation


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
    length = 2
    converged = False
    while not converged and length < 20:
        converged = True
        for i in range(1, length + 1):
            matchstate = states[-i]
            if states[-(i + length)] != matchstate:
                converged = False
        length += 1
    if converged:
        return length - 1
    else:
        raise Exception("Network did not Converge! Run for more itterations")


if __name__ == "__main__":

    # graph = (
    #    ClusteredGraphBuilder()
    #    .set_dgm_generation(nodes=100)
    #    .with_edge_degree_distribution(zipf, a=10)
    #    .with_triangle_degree_distribution(constant, k=0)
    #    .initialize_node_attributes_using({'states': list, 'next': str})
    #    .build()
    # )

    # graph = (
    #     DorogovtsevGoltsevMendesGraphBuilder()
    #     .set_dgm_generation(generation=6)
    #     .initialize_node_attributes_using({'states': list, 'next': str})
    #     .build()
    # )

    graph = (
        BarabasiAlbertGraphBuilder()
        .set_graph_size(nodes=145)
        .set_num_attachments(attachments=2)
        .initialize_node_attributes_using({'states': list, 'next': str})
        .build()
    )

    # if traingle is constant(3), convergence to all nodes at the same time

    #    graph = (
    #        ClusteredGraphBuilder()
    #        .set_dgm_generation(nodes=100)
    #        .with_edge_degree_distribution(constant, k=3)
    #        .with_triangle_degree_distribution(constant, k=3)
    #        .initialize_node_attributes_using({'states': list, 'next': str})
    #        .build()
    #    )

    # show you can get cycles of 6

    # graph = (
    #     ClusteredGraphBuilder()
    #     .set_dgm_generation(nodes=100)
    #     .with_edge_degree_distribution(uniform, low=1, high=6)
    #     .with_triangle_degree_distribution(uniform, low=0, high=2)
    #     .initialize_node_attributes_using({'states': list, 'next': str})
    #     .build()
    # )
    pos = nx.kamada_kawai_layout(graph)

    for node, data in graph.nodes(data=True):
        data['states'].append(random.choice(("rock", "paper", "scissors")))

    fig = plt.figure(frameon=False, figsize=(14, 8))

    def update(idx):
        fig.clear()
        nx.draw_networkx_edges(graph, pos=pos)

        nodes = {'rock': [], 'paper': [], 'scissors': []}
        for node, data in graph.nodes(data=True):
            nodes[data['states'][idx]].append(node)

        nx.draw_networkx_nodes(graph, pos=pos, nodelist=nodes['rock'], node_color='r', label='rock')
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=nodes['paper'], node_color='g', label='paper')
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=nodes['scissors'], node_color='b', label='scissors')

        plt.legend()

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

    for _ in range(50):
        for node, data in graph.nodes(data=True):
            neigh = graph[node]
            states = [graph.nodes[n]['states'][-1] for n in neigh]
            data["next"] = optimize_choice(states)

        for node, data in graph.nodes(data=True):
            data['states'].append(data['next'])
            data['next'] = ""

    # nx.draw_kamada_kawai(graph)
    anim = FuncAnimation(fig, update, frames=50, interval=400, repeat=False)
    anim.save('rps.mp4')
    plt.close(fig)

    conv_cycles = []
    for node, data in graph.nodes(data=True):
        data['conv_cycle'] = conv_cycle(data['states'])
    nodesets = {}
    for node, data in graph.nodes(data=True):
        if not (data['states'][-1], data['conv_cycle']) in nodesets:
            nodesets[(data['states'][-1], data['conv_cycle'])] = []
        nodesets[(data['states'][-1], data['conv_cycle'])].append(node)

    paper_counts = []
    rock_counts = []
    scissors_counts = []
    max_cycle_length = max([state_tuple[1] for state_tuple in nodesets])
    max_cycle_length = max_cycle_length / 3
    for i in range(1, int(max_cycle_length + 1)):
        cycle_length = i * 3
        cycle_exists = False
        for state_tuple in nodesets:
            if state_tuple[1] == cycle_length:
                cycle_exists = True
        if not cycle_exists:
            pass
        if ('paper', cycle_length) in nodesets:
            paper_counts.append(len(nodesets[('paper', cycle_length)]))
        else:
            paper_counts.append(0)
        if ('rock', cycle_length) in nodesets:
            rock_counts.append(len(nodesets[('rock', cycle_length)]))
        else:
            rock_counts.append(0)
        if ('scissors', cycle_length) in nodesets:
            scissors_counts.append(len(nodesets[('scissors', cycle_length)]))
        else:
            scissors_counts.append(0)
    indices = np.arange(max_cycle_length)
    plot1 = plt.bar(indices, paper_counts)
    plot2 = plt.bar(indices, rock_counts, bottom=paper_counts)
    plot3 = plt.bar(indices, scissors_counts,
                    bottom=[paper_counts[x] + rock_counts[x] for x in range(len(paper_counts))])
    plt.xticks(indices, (indices + 1) * 3)
    plt.xlabel("Cycle Length")
    plt.ylabel("Number of Nodes")
    plt.legend((plot1, plot2, plot3), ("paper", "rock", "scissors"))
    plt.show()
    plt.close()

    degreesets = {}
    max_deg = 0
    for index in nodesets:
        degreesets[index] = []
        for node in nodesets[index]:
            degreesets[index].append((node, graph.degree[node]))
            if graph.degree[node] > max_deg:
                max_deg = graph.degree[node]
