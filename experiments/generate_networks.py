import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tensorpowerflow import generate_network

#%%
fig, ax = plt.subplots(10, 10, figsize=(10, 10))
plt.subplots_adjust(wspace=0.25, hspace=0.25)
ax_ = ax.flatten()
for _ax in ax_:
    child, nodes = np.random.randint(2, 5), np.random.randint(10, 100)
    print(f"Child: {child}, Nodes: {nodes}")
    G = nx.full_rary_tree(child, nodes)
    nx.draw_kamada_kawai(G, node_size=4, font_size='small', ax=_ax)
    _ax.set_title(f"n: {nodes} c: {child}", fontsize="xx-small", y=0.85)
    assert nx.is_connected(G)

#%%
NODES = 200
nodes_frame, lines_frame = generate_network(NODES, child=3, plot_graph=True, load_factor=2, line_factor=3)

