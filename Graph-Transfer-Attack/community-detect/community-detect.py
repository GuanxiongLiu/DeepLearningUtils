import community
import networkx as nx
import argparse
import sys
import numpy as np






if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="Community Detection")
    parser.add_argument('--edge', nargs='?', help='Input edge list path')
    parser.add_argument('--save', nargs='?', help='Input saving path')
    args = parser.parse_args()
    # load edge list
    matrix = np.loadtxt(args.edge, delimiter=' ')
    # load to graph
    G = nx.Graph()
    for r in range(matrix.shape[0]):
        G.add_edge(matrix[r,0], matrix[r,1])
    # compute the best partition
    partition = community.best_partition(G)
    # get results
    res = []
    for cmty in set(partition.values()):
        for nodes in partition.keys():
            if partition[nodes] == cmty:
                res.append([nodes, cmty])
    res = np.array(res)
    # save
    np.savetxt(args.save, res)