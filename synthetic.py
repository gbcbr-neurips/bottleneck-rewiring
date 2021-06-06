import networkx as nx
import numpy as np
from utils import adj

def get_K_neighbors(G, node, K):
    return [k for k,v in nx.single_source_shortest_path_length(G, node, cutoff=K).items() if v==K]

def add_value_to_nodes(G, value_name, values):
    for i in sorted(G.nodes):
        G.nodes[i][value_name] = values[i]

def average_value_by_K_neighbors(G, node, K, value_name, average_name, make_binary=False):
    values = [G.nodes[n][value_name] for n in get_K_neighbors(G, node, K) + [node]]
    if make_binary:
        G.nodes[node][average_name] = np.mean(values) > 0
    else:
        G.nodes[node][average_name] = np.mean(values)

def generate_graph_barbell(N2, p, K, correlation=0, make_binary=False):
    G = nx.fast_gnp_random_graph(N2,p)
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    G1 = nx.relabel_nodes(G, {list(G.nodes)[i]:i for i in range(len(G.nodes))})
    G = nx.fast_gnp_random_graph(N2,p)
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    N1 = len(G1.nodes)
    G2 = nx.relabel_nodes(G, {list(G.nodes)[i-N1]:i for i in range(N1, N1+len(G.nodes))})
    G = nx.compose(G1, G2)
    G.add_edge(0,len(G1.nodes))
    adj_wo_bridge = adj(G)
    adj_wo_bridge[0, len(G1.nodes)] = adj_wo_bridge[len(G1.nodes), 0] = 0
    add_value_to_nodes(G, 'x', np.random.multivariate_normal(
        np.zeros(len(G.nodes)),
        np.identity(len(G.nodes)) + correlation*adj_wo_bridge
    ))
    for n in G.nodes:
        average_value_by_K_neighbors(G, n, K, 'x', 'y', make_binary=make_binary)
    return G
