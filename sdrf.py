import networkx as nx
import numpy as np

from curvature import unbiased_forman_parallel

softmax = lambda x: np.exp(x)/sum(np.exp(x))

def focused_stochastic_ricci_step(G, A, N, curvature_fn, target_curvature=0,
                          prioritise_betweenness=False, consider_positivity=True):
    curvatures, AE, no_of_edges, mapping, ijs = curvature_fn(G, A, N, target_curvature)
    
    candidates = {}
    x = AE*target_curvature - curvatures
    if not consider_positivity:
        x = (abs(x) + x) / 2
    mses = (x**2).sum(axis=2).sum(axis=1)/no_of_edges
    mses = mses[0] - mses
    
    if prioritise_betweenness:
        betweenness = {k:v for (k,v) in nx.algorithms.centrality.edge_betweenness_centrality(G).items() if k in ijs}
        betweenness = {k: v for k, v in sorted(betweenness.items(), key=lambda item: -item[1])}
        ijs = list(betweenness.keys())
    else:
        ijs = [(i,j) for (i,j) in ijs if i < j]
    
    for i, j in ijs:
        for x in list(G.neighbors(i)) + [i]:
            for y in list(G.neighbors(j)) + [j]:
                x_, y_ = (x,y) if x<y else (y,x)
                if x_ != y_ and not A[x_,y_] and not (x_,y_) in candidates and mses[mapping[(x_,y_)]] > 0:
                        candidates[(x_,y_)] = mses[mapping[(x_,y_)]]
        if prioritise_betweenness and len(candidates) > 0:
            break
            
    return candidates


STR_TO_CURVATURE_MAP = {
    # 'uma_forman': uma_forman_parallel,
    'unbiased_forman': unbiased_forman_parallel,
}
STR_TO_RICCI_STEP = {
    'fsdrf': focused_stochastic_ricci_step,
}

def stochastic_discrete_ricci_flow(
    G,
    curvature_fn,
    target_curvature=0,
    scaling=2,
    only_allow_positive_actions=True,
    prioritise_betweenness=False,
    consider_positivity=True,
    ricci_step='fsdrf',
    max_steps=None
):
    A, N = nx.to_numpy_array(G), len(G.nodes)
    added_edges = []
    removed_edges = []
    graphs = [G]

    if isinstance(curvature_fn, str):
        curvature_fn = STR_TO_CURVATURE_MAP[curvature_fn]

    if isinstance(ricci_step, str):
        ricci_step = STR_TO_RICCI_STEP[ricci_step]
    
    if max_steps is None:
        i = -1
    elif max_steps < 1:
        i = np.ceil((A.sum() / 2) * max_steps)
    else:
        i = max_steps
    
    while i != 0:
        scores = ricci_step(
            graphs[-1],
            A,
            N,
            curvature_fn,
            target_curvature=target_curvature,
            prioritise_betweenness=prioritise_betweenness,
            consider_positivity=consider_positivity
        )
        if only_allow_positive_actions:
            scores = {k:v*scaling for k,v in scores.items() if v > 0}
        scores_keys = list(scores.keys())
        scores_values = np.array(list(scores.values()))
        if np.any(scores_values > 0):
            new_graph = graphs[-1].copy()
            if len(scores_keys) > 1:
                x,y = scores_keys[
                    np.random.choice(
                        range(len(scores_keys)),
                        p=softmax(scores_values)
                    )
                ]
            else:
                x,y = scores_keys[0]
            if new_graph.has_edge(x,y):
                new_graph.remove_edge(x,y)
                A[x,y] = A[y,x] = 0
                removed_edges.append((x,y))
            else:
                new_graph.add_edge(x,y)
                A[x,y] = A[y,x] = 1
                added_edges.append((x,y))
            i -= 1
            graphs.append(new_graph)
        else:
            break
    
    return graphs
