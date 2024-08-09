import numpy as np
import jax.numpy as jnp
import jax
import networkx as nx

def init_graph_from_graph(G: nx.MultiDiGraph, mu: float, std: float=0, seed=1234):
    node_to_id = {val: i for i, val in enumerate(list(G.nodes()))}

    I = np.zeros((len(node_to_id), len(node_to_id)))
    A = np.zeros_like(I)
    L = np.zeros_like(I)
    rho = np.zeros_like(I)

    key = jax.random.PRNGKey(seed)

    for e_i,e_j, id in G.edges:
        key, subkey =  jax.random.split(key)
        
        i,j = node_to_id[e_i], node_to_id[e_j]
        rho[i,j]=std*jax.random.normal(subkey)+mu
        A[i,j]=1.
        I[i,j]=1. # set depending highway type (osm)
        L[i,j]=G.edges[e_i, e_j, id]['length']

         

    return jnp.array(rho), jnp.array(A), jnp.array(I), jnp.array(L)