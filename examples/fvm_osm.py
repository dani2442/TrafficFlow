import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import jax
import jax.numpy as jnp
import numpy as np

from trafficflow.generate import init_graph_from_graph
from trafficflow.fvm import fvm_graph
from trafficflow import rho_max
from trafficflow.utils import plot_osm_graph


def main():
    place_name = "Piedmont, California, USA"
    G = ox.graph_from_place(place_name, network_type='drive', simplify=False)  

    T_sim = 6
    N_t = 300
    dt = T_sim/N_t

    rho, A, I, L = init_graph_from_graph(G, rho_max/4, rho_max/4)

    rho_checkpoints = []
    for t_i in range(N_t+1):
        rho = fvm_graph(rho, A, I, L, dt)
        if t_i%50 == 0:
            rho_checkpoints += [rho]

    f, axs = plt.subplots(2, len(rho_checkpoints)//2, figsize=(15,5))
    for i, ax in enumerate(axs.reshape(-1)):
        plot_osm_graph(G, rho_checkpoints[i], ax=ax)

    f.savefig('images/fvm_osm.png', dpi=300)
    plt.plot()

if __name__ == '__main__':
    main()