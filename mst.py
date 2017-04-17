import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import timeit

#TODO: Determine runtimes for algorithms

def kruskels(graph):
    tree = []
    # Sort the edge matrix by cost
    graph = np.array(graph)
    graph = graph[graph[:, 2].argsort()]
    graph = tuple(map(tuple, graph))
    # print(graph)
    edges = 0
    # For each edge, add to the graph, and determine if it creates a cycle
    for edge in graph:
        # Max number of edges in a tree = number of edges - 1.
        if edges < 999 - 1:
            trcopy = tree.copy()
            # print(edge)
            trcopy.append(edge)
            if cyclecheck(trcopy) == False:
                tree = trcopy
                edges = edges + 1
                #G = nx.Graph()
                #G.add_weighted_edges_from(tree)
                #nx.draw(G)
                #plt.show()
    return tree
    

# Test function
def prims(graph):
    tree = nx.Graph()
    G = nx.Graph()
    graph = tuple(map(tuple, graph))
    G.add_weighted_edges_from(graph)
    # Start with the first node in the graph
    tree.add_node(G.nodes()[0])
    edges = 0
    # Find the least cost edge in the graph
    # Find the minimum weight edge connected to the nodes in the new graph
    edges = 0
    while (edges < len(G.nodes()) - 1):
        minweight = 999999999
        minedge = (-1, -1)
        for node in tree.nodes():
            for neighbor in G.neighbors(node):
                if G.get_edge_data(node, neighbor)['weight'] < minweight:
                    if (neighbor in tree.nodes()) == False:
                        minweight = G.get_edge_data(node, neighbor)['weight']
                        minedge = (node, neighbor, minweight)
        # Add the edge to the new graph
        tree.add_weighted_edges_from([minedge])
        edges = edges + 1
    return True

# Tests for a cycle in a graph.  
def cyclecheck(graph):
    G = nx.Graph()
    G.add_weighted_edges_from(graph)
    try:
        nx.find_cycle(G)
        # print("Cycle found, discarding")
        return True
    except nx.exception.NetworkXNoCycle:
        return False

# Prune disconnected nodes.
def prune_tree(cur_graph):
    if not nx.is_connected(cur_graph):
        # get a list of unconnected networks
        sub_graphs = nx.connected_component_subgraphs(cur_graph)

        subgraphs = list(sub_graphs)
        main_graph = subgraphs[0]

        # find the largest network in that list
        for sg in sub_graphs:
            print(sg)
            if len(sg.nodes()) > len(main_graph.nodes()):
                main_graph = sg

        return main_graph

# Matrix format - [edge1, edge2, weight]
'''testcase = np.matrix([
    [0, 1, 2],
    [0, 3, 6],
    [0, 4, 5],
    [1, 2, 3],
    [1, 4, 1],
    [1, 5, 1],
    [2, 3, 5],
    [3, 4, 4],
    [4, 5, 1]
])'''

'''testcase = np.matrix([
    [0, 1, 1],
    [0, 2, 3],
    [0, 4, 4],
    [0, 5, 3],
    [1, 3, 1],
    [2, 3, 2],
    [4, 5, 1]
])'''

# testcase = np.loadtxt("gr_dense_1000.txt")

# Timeit will determine runtime

# Kruskels testcase
print("Kruskels")
start = timeit.default_timer()
graph = kruskels(testcase)
stop = timeit.default_timer()
runtime_kruskels = stop - start

# Print all edges in the graph, then sum all their weights and return that as well 

G = nx.Graph()
G.add_weighted_edges_from(graph)    
edgesum = G.size(weight='weight')
print("Total weight is {0}".format(edgesum))

print(list(G.edges_iter(data='weight', default=1)))
print("Ran in {0} seconds".format(runtime_kruskels))

# Repeat for prims
print("Prims")
start = timeit.default_timer()
graph = kruskels(testcase)
stop = timeit.default_timer()
runtime_kruskels = stop - start

# Print all edges in the graph, then sum all their weights and return that as well 

G = nx.Graph()
G.add_weighted_edges_from(graph)    
edgesum = G.size(weight='weight')
print("Total weight is {0}".format(edgesum))
print(list(G.edges_iter(data='weight', default=1)))
print("Ran in {0} seconds".format(runtime_kruskels))

#nx.draw(G)
#plt.show()