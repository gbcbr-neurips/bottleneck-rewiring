import networkx as nx
import numpy as np

def slanted_adj(A):
    b = np.zeros((A.shape[0], A.shape[1]*A.shape[1]))
    b[:, ::A.shape[1]+1] = A
    return np.swapaxes(b.reshape(A.shape[0], A.shape[1], A.shape[1]), 0, 1)

def slanted_adj_4D(A):
    b = np.zeros((A.shape[0], A.shape[1], A.shape[2]*A.shape[2]))
    b[:, :, ::A.shape[2]+1] = A
    return np.swapaxes(b.reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[2]), 1, 2)

def get_sharps_and_lambdas(A, N):
    Ax = A.reshape(N,N,-1)
    Ay = A.reshape(N,-1,N)
    T = (A.dot(A * (1 - Ax)) * (1 - slanted_adj(A))) - 1
    four_cycles = T * Ax * (1 - Ay)
    sharps = (four_cycles > 0).sum(axis=0)
    lambdas = four_cycles.max(axis=0)
    lambdas = np.maximum(lambdas, lambdas.T)
    return sharps, lambdas

def get_sharps_and_lambdas_3D(AE, N):
    M = len(AE)
    Ax = AE.reshape(M,N,N,-1)
    Ay = AE.reshape(M,N,-1,N)
    T = (np.einsum('...ij,...kjm->...ikm', AE, np.einsum('...jk,...ijk->...ijk', AE, (1-Ax))) * (1 - slanted_adj_4D(AE))) - 1
    four_cycles = T * Ax * (1 - Ay)
    sharps = (four_cycles > 0).sum(axis=1)
    lambdas = four_cycles.max(axis=1)
    lambdas = np.maximum(lambdas, np.einsum('ijk->ikj', lambdas))
    return sharps, lambdas

def unbiased_forman(A, N):
    degrees = A.sum(axis=0)
    degrees_reshaped = degrees.reshape((-1,N))
    degrees_reshaped_T = degrees.reshape((N,-1))
    degrees_max = np.maximum(degrees_reshaped, degrees_reshaped_T)
    degrees_min = np.minimum(degrees_reshaped, degrees_reshaped_T)
    A2 = np.matmul(A,A)
    sharps, lambdas = get_sharps_and_lambdas(A,N)
    return (np.round((2 / degrees_reshaped) + (2 / degrees_reshaped_T) - 2 + 2*A2/degrees_max + A2/degrees_min \
        + np.nan_to_num((sharps + sharps.T)/(lambdas*degrees_max)),2)) * A

def unbiased_forman_parallel(G, A, N, target_curvature, return_extra=True):
    base_no_of_edges = A.sum()
    no_of_edges = np.array([base_no_of_edges])
    
    C = unbiased_forman(A,N)
    minC = np.min(C)
    if minC >= target_curvature:
        if return_extra:
            return (C*A).reshape(1,N,N), A.reshape(1,N,N), no_of_edges, {}, []
        else:
            return (C*A).reshape(1,N,N)
    
    minima = np.argwhere(C == minC)
    ijs = [(minimum[0],minimum[1]) for minimum in minima]
    ijs = [(i,j) for (i,j) in ijs if i < j]
    E = np.zeros((1,N,N))
    l = 1
    mapping = {}
    
    for i, j in ijs:
        for x in list(G.neighbors(i)) + [i]:
            for y in list(G.neighbors(j)) + [j]:
                x_, y_ = (x,y) if x<y else (y,x)
                if x_ != y_ and not A[x_,y_] and not (x_,y_) in mapping:
                    E = np.append(E, np.zeros((1,N,N)), axis=0)
                    E[l,x,y] = E[l,y,x] = 1
                    no_of_edges = np.append(no_of_edges, [base_no_of_edges+1])
                    mapping[(x_,y_)] = l
                    l += 1
    NC2 = l
    
    AE = A+E
    AE2 = np.matmul(AE,AE)
    degrees = np.diagonal(AE2, axis1=1, axis2=2)
    degrees_reshaped = degrees.reshape((NC2,-1,N))
    degrees_reshaped_T = degrees.reshape((NC2,N,-1))
    degrees_max = np.maximum(degrees_reshaped, degrees_reshaped_T)
    degrees_min = np.minimum(degrees_reshaped, degrees_reshaped_T)
    sharps, lambdas = get_sharps_and_lambdas_3D(AE,N)
    curvatures = (2 / degrees_reshaped) + (2 / degrees_reshaped_T) - 2 + 2*AE2/degrees_max + AE2/degrees_min + np.nan_to_num((sharps + np.einsum('ijk->ikj', sharps))/(lambdas*degrees_max))
    if return_extra:
        return curvatures*AE, AE, no_of_edges, mapping, ijs
    return curvatures*AE

def give_graph_ricci(G, curvatures=None):
    if curvatures is None:
        A = nx.to_numpy_array(G)
        N = len(G.nodes)
        curvatures = unbiased_forman(A, N)
    for x,y in G.edges:
        G.edges[x,y]['ricci'] = curvatures[x,y]
