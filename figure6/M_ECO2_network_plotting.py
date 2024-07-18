# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import loadmat

A=loadmat("../Networks/ECO2.mat")['A']
G=nx.from_numpy_matrix(A)
G=nx.subgraph(G, max(nx.connected_components(G)))
G=nx.convert_node_labels_to_integers(G)
A=nx.to_numpy_matrix(G)
A=(A>0.3)
G=nx.from_numpy_matrix(A)
G=nx.subgraph(G, max(nx.connected_components(G)))
G=nx.convert_node_labels_to_integers(G)
A=nx.to_numpy_matrix(G)
G=nx.subgraph(G,range(100,150))
G = nx.DiGraph(G)

fig=plt.figure(figsize=(10, 10))
ax=fig.add_subplot(111)
pos = nx.circular_layout(G)
nodes = nx.draw_networkx_nodes(G, pos, node_size=50, node_color='#328CA0')
for edge in G.edges:
    if edge[0] > edge[1]+1:
        if abs(edge[0]- edge[1]) <10:
            nx.draw_networkx_edges(G, pos, edge_color='#328CA0', width=1, alpha=0.35,edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {0.3}')
        else:
            nx.draw_networkx_edges(G, pos, edge_color='#328CA0', width=1, alpha=0.35,edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.3}')
    if edge[0] ==   edge[1]+1:
        nx.draw_networkx_edges(G, pos, edge_color='#328CA0', width=1, alpha=0.35,edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.1}')

ax.set_aspect('equal')
ax.set_axis_off()
plt.savefig('M_ECO2_network.pdf',dpi = 300)
plt.show()