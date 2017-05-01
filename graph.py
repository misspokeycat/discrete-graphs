import networkx as nx
import numpy
import matplotlib.pyplot as plt

A = numpy.matrix([[0, 1, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0]])

G = nx.from_numpy_matrix(A)
print("Eccentricity is {0}".format(nx.eccentricity(G)))
print("Diameter is {0}".format(nx.diameter(G)))
#nx.draw(G)
#plt.show()