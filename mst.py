''' mst.py
    Minimum spanning tree algorithm implementations.
'''

# Imports
import numpy as np # numpy is a great math/array library
import networkx as nx # netowrkx is a graph library, very useful for manipulating graphs
import matplotlib.pyplot as plt # commented out, but used for displaying the graph when finished
import timeit # used for determining runtime of algorithms

# Kruskels algorithm
# Perfomrs the kruskels minimum spanning tree algorithm on a numpy matrix or edge list.
# Returns: edge tuple list
def kruskels(graph):
    tree = []
    # Sort the edge matrix by cost
    graph = np.array(graph)
    graph = graph[graph[:, 2].argsort()]
    graph = tuple(map(tuple, graph))
    edges = 0
    # For each edge, add to the graph, and determine if it creates a cycle
    for edge in graph:
        # Max number of edges in a tree = number of edges - 1.
        if edges < 999 - 1:
            trcopy = tree.copy()
            trcopy.append(edge)
            if cyclecheck(trcopy) == False:
                tree = trcopy
                edges = edges + 1
    return tree
    

# Test function
def prims(graph):
    # We'll be manipulating our graphs using networkx.
    tree = nx.Graph()  # The tree we'll return
    G = nx.Graph() # The "source" tree
    # Convert the numpy array into a tuple list.
    # This will look like ((0,1,2),(0,2,3)) etc.
    graph = np.array(graph)
    graph = tuple(map(tuple, graph))
    # Create our graph from the edge weight list.
    G.add_weighted_edges_from(graph)
    # Add the first vertex (node) in the source graph to our new graph.
    tree.add_node(G.nodes()[0])
    # Find the minimum weight edge connected to the nodes in the new graph,
    # and add it to to the new graph.
    # Edge counter.  Since the maximum number of edges in a min. spanning tree 
    # is number of vertexes - 1, we can stop when we reach that number of edges.
    edges = 0 
    while (edges < len(G.nodes()) - 1):
        minweight = 999999999 # smallest weight
        minedge = (-1, -1) # edge (not in graph) with smallest weight
        for node in tree.nodes():  # iterate over all nodes in new tree
            for neighbor in G.neighbors(node): # iterate over all edges connected to nodes
                if G.get_edge_data(node, neighbor)['weight'] < minweight: # We want the node if it's the smallest...
                    if (neighbor in tree.nodes()) == False: # ...and if we don't have the target node in in our new tree.
                        minweight = G.get_edge_data(node, neighbor)['weight']
                        minedge = (node, neighbor, minweight)
        # Add the edge to the new graph
        tree.add_weighted_edges_from([minedge])
        edges = edges + 1
        # This returns our tree as a list of tuples.
    return list(tree.edges_iter(data='weight', default=1))

# Tests for a cycle in a graph.  
def cyclecheck(graph):
    # Create a networkx graph from an input edge tuple list.
    G = nx.Graph()
    G.add_weighted_edges_from(graph)
    # Almost cheating, but we can use the nx.find_cycle() function to check for cycles.
    # If there is no cycle, networkx throws an exception, and we return false.
    try:
        nx.find_cycle(G)
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

testcase = np.matrix([
    [0, 1, 1],
    [0, 2, 3],
    [0, 4, 4],
    [0, 5, 3],
    [1, 3, 1],
    [2, 3, 2],
    [4, 5, 1]
])

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
testcase = np.matrix([
    [0, 1, 1],
    [0, 2, 3],
    [0, 4, 4],
    [0, 5, 3],
    [1, 3, 1],
    [2, 3, 2],
    [4, 5, 1]
])

print("Prims")
start = timeit.default_timer()
graph = prims(testcase)
stop = timeit.default_timer()
runtime_prims = stop - start

# Print all edges in the graph, then sum all their weights and return that as well 

G = nx.Graph()
G.add_weighted_edges_from(graph)
edgesum = G.size(weight='weight')
print("Total weight is {0}".format(edgesum))
print(list(G.edges_iter(data='weight', default=1)))
print("Ran in {0} seconds".format(runtime_prims))

#nx.draw(G)
#plt.show()