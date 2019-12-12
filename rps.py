import csv
import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from distributions import constant, uniform, zipf
from factories import *
from matplotlib.animation import FuncAnimation
from scipy.stats import poisson


def optimize_choice(states: list):
    mode_state = Counter(states).most_common(1)[0][0].lower()
    return {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock",
    }[mode_state]


def is_losing(this: str, other: str):
    return {
        "rock": other.lower() == "paper",
        "paper": other.lower() == "scissors",
        "scissors": other.lower() == "rock",
    }[this.lower()]


def is_winning(this: str, other: str):
    return {
        "rock": other.lower() == "scissors",
        "paper": other.lower() == "rock",
        "scissors": other.lower() == "paper",
    }[this.lower()]


def is_tied(this: str, other: str):
    return {
        "rock": other.lower() == "rock",
        "paper": other.lower() == "paper",
        "scissors": other.lower() == "scissors",
    }[this.lower()]


def get_outcome(this: str, other: str):
    if is_winning(this, other):
        return 'win'
    elif is_tied(this, other):
        return 'tie'
    elif is_losing(this, other):
        return 'lose'
    else:
        return None


def identify_convergence(states: list):
    length = 2
    converged = False
    while not converged and length <= 21:
        converged = True
        for i in range(1, length + 1):
            state_to_match = states[-i]
            previous_state_idx = -(length + i)
            if states[previous_state_idx] != state_to_match:
                converged = False
        length += 1
    if converged:
        return length - 1
    else:
        raise Exception("Network did not Converge! Run for more iterations")


def create_convergence_plot(step):
    for node, data in graph.nodes(data=True):
        data["convergence"] = identify_convergence(data["states"])

    nodesets = {}
    for node, data in graph.nodes(data=True):
        key = (data["states"][-1], data["convergence"])
        if key not in nodesets:
            nodesets[key] = []
        nodesets[key].append(node)

    counts = {
        "rock": [],
        "paper": [],
        "scissors": [],
    }
    max_cycle_length = max([state_tuple[1] for state_tuple in nodesets]) // 3
    for i in range(1, max_cycle_length + 1):
        cycle_length = i * 3
        cycle_exists = any(state_tuple[1] == cycle_length for state_tuple in nodesets)
        if not cycle_exists:
            continue

        counts["paper"].append(
            len(nodesets[("paper", cycle_length)])
            if ("paper", cycle_length) in nodesets
            else 0
        )
        counts["rock"].append(
            len(nodesets[("rock", cycle_length)])
            if ("rock", cycle_length) in nodesets
            else 0
        )
        counts["scissors"].append(
            len(nodesets[("scissors", cycle_length)])
            if ("scissors", cycle_length) in nodesets
            else 0
        )

    indices = np.arange(max_cycle_length)
    fig, ax = plt.subplots()
    ax.bar(indices, counts["paper"], label="paper")
    ax.bar(indices, counts["rock"], label="rock", bottom=counts["paper"])
    ax.bar(
        indices,
        counts["scissors"],
        label="scissors",
        bottom=[sum(t) for t in zip(counts["paper"], counts["rock"])],
    )

    ax.set_xticks(indices)
    ax.set_xticklabels((indices + 1) * 3)
    ax.set_xlabel("Cycle Length")
    ax.set_ylabel("Number of Nodes")
    ax.legend()
    plt.savefig(f"convergence-{step}.png", bbox_inches='tight')
    fig.clear()
    # fig.show()


def create_degree_dist_plot():
    nodesets = {}
    for node, data in graph.nodes(data=True):
        key = (data["states"][-1], data["convergence"])
        if key not in nodesets:
            nodesets[key] = []
        nodesets[key].append(node)

    fig, ax = plt.subplots()
    degree_list = sorted(graph.degree[n] for n in graph.nodes)
    mean_degree = sum(degree_list) / len(degree_list)
    fit = poisson.pmf(degree_list, mean_degree)
    ax.plot(degree_list, fit, "-o")
    ax.hist(degree_list, normed=True)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Proportion of Nodes")
    plt.savefig('degree_dist.png', bbox_inches='tight')


def save_reward_distribution(filename):
    rows = []
    for n, states in graph.nodes(data='states'):
        dist = {'id': n, 'win': 0, 'tie': 0, 'lose': 0}
        this_state = states[-1]
        neighbors = graph[n]
        for neigh in neighbors:
            neigh_state = graph.nodes[neigh]['states'][-1]
            dist[get_outcome(this_state, neigh_state)] += 1
        rows.append(dist)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('id', 'win', 'tie', 'lose'))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":

    graph = (
        BarabasiAlbertGraphBuilder()
        .set_graph_size(nodes=145)
        .set_num_attachments(attachments=2)
        .initialize_node_attributes_using({"states": list, "next": str})
        .build()
    )

    for node, data in graph.nodes(data=True):
        data["states"].append(random.choice(("rock", "paper", "scissors")))

    fig, ax = plt.subplots(frameon=False, figsize=(14, 8))
    pos = nx.kamada_kawai_layout(graph)

    def run():
        steps = 100
        rewires = 4

        def simulate(timesteps):
            for s in range(timesteps):
                for n, d in graph.nodes(data=True):
                    neigh = graph[n]
                    d["next"] = (
                        optimize_choice([graph.nodes[n]["states"][-1] for n in neigh])
                        if neigh
                        else d["states"][-1]
                    )

                for n, d in graph.nodes(data=True):
                    d["states"].append(d["next"])
                    d["next"] = ""

                yield s

        def rewire():
            for n, s in graph.nodes(data='states'):
                this_state = s[-1]
                winning_neighbors = [
                    n
                    for n in graph[n]
                    if is_losing(this_state, graph.nodes[n]["states"][-1])
                ]
                if not winning_neighbors:
                    continue
                disconnect_from = random.choice(winning_neighbors)

                beatable_nodes = {u for u, st in graph.nodes(data='states') if is_winning(this_state, st[-1])}
                tieable_nodes = {u for u, st in graph.nodes(data='states') if is_tied(this_state, st[-1])}
                possible_new_neighbors = list((beatable_nodes if beatable_nodes else tieable_nodes)
                                              - set(graph[n])
                                              - {n})
                if possible_new_neighbors:
                    graph.remove_edge(n, disconnect_from)
                    graph.add_edge(n, random.choice(possible_new_neighbors))

        for step in simulate(steps):
            if step != 0 and step % (steps // rewires) == 0:
                rewire()
                create_convergence_plot(step)

            yield step

    def update(idx):
        ax.clear()
        nx.draw_networkx_edges(graph, pos=pos)

        states = {"rock": [], "paper": [], "scissors": []}
        for node, data in graph.nodes(data=True):
            state = data["states"][idx]
            states[state].append(node)

        nx.draw_networkx_nodes(
            graph, pos=pos, nodelist=states["rock"], node_color="r", label="rock"
        )
        nx.draw_networkx_nodes(
            graph, pos=pos, nodelist=states["paper"], node_color="g", label="paper"
        )
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=states["scissors"],
            node_color="b",
            label="scissors",
        )

        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    save_reward_distribution('before.csv')
    anim = FuncAnimation(fig, func=update, frames=run, interval=300, repeat=False)
    anim.save("rps.mp4")
    save_reward_distribution('after.csv')

    create_degree_dist_plot()
