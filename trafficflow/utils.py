import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import osmnx as ox
import numpy as np


def plot_osm_graph(G, rho, ax):
    node_to_id = {val: i for i, val in enumerate(list(G.nodes()))}

    weights = [rho[node_to_id[u],node_to_id[v]] for u,v in G.edges()]
    # Normalize the weights for colormap
    norm = plt.Normalize(min(weights), vmax=max(weights))
    
    cmap = plt.cm.viridis
    norm = Normalize(vmin=np.min(weights), vmax=np.max(weights))
    
    fig, ax = ox.plot_graph(G, edge_color=cmap(norm(weights)), node_size=0, edge_linewidth=0.8, bgcolor = 'white', show=False, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation='vertical')