import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
        .with_triangle_degree_distribution(uniform, low=0, high=2)
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

    conv_cycles=[]
    for node, data in graph.nodes(data=True):
        data['conv_cycle'] = conv_cycle(data['states'])
    nodesets={}
    for node, data in graph.nodes(data=True):
        if not (data['states'][-1],data['conv_cycle']) in nodesets:
            nodesets[(data['states'][-1],data['conv_cycle'])]=[]
        nodesets[(data['states'][-1],data['conv_cycle'])].append(node)
    
    paper_counts=[]
    rock_counts=[]
    scissors_counts=[]
    max_cycle_length = max([state_tuple[1] for state_tuple in nodesets])
    max_cycle_length = max_cycle_length/3
    for i in range(1,int(max_cycle_length+1)):
        cycle_length = i*3
        cycle_exists=False
        for state_tuple in nodesets:
            if state_tuple[1]==cycle_length:
                cycle_exists=True
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
    indices = np.arange((max_cycle_length/3)+1)
    plot1 = plt.bar(indices, paper_counts)
    plot2 = plt.bar(indices, rock_counts, bottom=paper_counts)
    plot3 = plt.bar(indices, scissors_counts, bottom=[sum(x) for x in zip(rock_counts, paper_counts)])
    plt.xticks(indices, (indices+1)*3)
    plt.xlabel("Cycle Length")
    plt.ylabel("Number of Nodes")
    plt.legend((plot1, plot2, plot3),("paper","rock","scissors"))
    plt.show()
    
    
    degreesets={}
    max_deg=0
    for index in nodesets:
        degreesets[index]=[]
        for node in nodesets[index]:
            degreesets[index].append((node, graph.degree[node]))
            if graph.degree[node]>max_deg:
                max_deg = graph.degree[node]
    
