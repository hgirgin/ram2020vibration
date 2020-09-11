import networkx as nx
import numpy as np

def calc_dist(traj):
    dist = 0
    for i in range(len(traj)-1):
        dist += np.linalg.norm(traj[i]-traj[i+1])
    return dist

def extract_traj(graph, path):
    total_disp = 0
    traj = []
    for i,idx in enumerate(path):
        traj += [graph.nodes[idx]['value']]
        if i != 0:
            print(graph.edges()[path[i-1], path[i]])
            total_disp += graph.edges()[path[i-1], path[i]]['disp']
    traj = np.array(traj)

    return traj, total_disp

def from_coordinate_list(coordinates, distfn = None):
    num_nodes = len(coordinates)
    graph = nx.Graph()
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            graph.add_node(i, value=coordinates[i])
            graph.add_node(j, value=coordinates[j])
            dist = distfn(coordinates[i], coordinates[j])
            graph.add_edge(i, j, weight=dist)
    return graph

def construct_cgraph_old(qs, disps = None, w_disp = 1.):
    cgraph = nx.Graph()
    nodes = []
    last_node = 0

    cur_indices = [0]
    cgraph.add_node(last_node, value=qs[0][0])
    for i in range(len(qs[:-1])):
        print(i)
        q_cur = qs[i]
        q_next = qs[i+1]
        next_indices = np.arange(last_node+1, last_node+1+len(q_next))
        for k in range(len(q_next)):
            last_node += 1
            cgraph.add_node(last_node, value = q_next[k])
            for j in range(len(q_cur)):
                dist = np.linalg.norm(q_cur[j]-q_next[k])
                if disps is None:
                    disp = 0
                else:
                    disp = disps[i+1][k]
                weight = dist + w_disp*disp
                cgraph.add_edge(last_node, cur_indices[j], weight=weight, disp = disp)
        cur_indices = next_indices
    return cgraph

def construct_cgraph(qs, disps = None, w_disp = 1.):
    cgraph = nx.Graph()
    last_node = 0

    cur_indices = [0]
    cgraph.add_node(last_node, value=qs[0][0])
    for i in range(len(qs[:-1])):
        print(i)
        q_cur,q_next = qs[i], qs[i+1]
        next_indices = np.arange(last_node+1, last_node+1+len(q_next))
        for k in range(len(q_next)):
            last_node += 1
            cgraph.add_node(last_node, value = q_next[k])
            dists = np.linalg.norm(q_cur-q_next[k],axis=1)
            if disps is None: disps = np.zeros(len(dists))
            weights = dists + w_disp*disps[i]
            edges = [(last_node, cur_indices[j], {'weight':weights[j], 'disp':disps[i][j]}) for j in range(len(q_cur))]
            cgraph.add_edges_from(edges)
        cur_indices = next_indices
    return cgraph


def euclidean_metric(c1, c2):
    return np.linalg.norm(c1-c2)
    
def reverse(a):
    return a[::-1]
    
def two_opt(graph, weight='weight'):
    num_nodes = graph.number_of_nodes()
    #tour = graph.nodes()
    tour = np.arange(num_nodes)
    # min_cost = compute_tour_cost(graph, tour)
    start_again = True
    while start_again:
        start_again = False
        for i in range(0, num_nodes-1):
            for k in range(i+1, num_nodes):
                # 2-opt swap
                a, b = tour[i-1], tour[i]

                c, d = tour[k], tour[(k+1)%num_nodes]
                if (a == c) or (b == d):
                    continue
                ab_cd_dist = graph.edges[a,b][weight] + graph.edges[c,d][weight]
                ac_bd_dist = graph.edges[a,c][weight] + graph.edges[b,d][weight]
                if ab_cd_dist > ac_bd_dist:
                    tour[i:k+1] = reverse(tour[i:k+1])
                    start_again = True
                if start_again:
                    break
            if start_again:
                break
    return tour