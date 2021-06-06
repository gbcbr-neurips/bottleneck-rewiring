import networkx as nx
import numpy as np

def adj(G):
    return np.asarray(nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense())
