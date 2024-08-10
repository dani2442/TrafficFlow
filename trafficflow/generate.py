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


def init_graph_2d(rho_max, N_x, N_t, x_0, x_N):
    xs = jnp.linspace(x_0, x_N, N_x)
    rho_0x = rho_max/4 + jnp.exp(-jnp.square((xs - x_N/2)/20))*rho_max/2
    rho_t0 = 0#rho_max/4 + jnp.zeros(N_t) 
    rho_t1 = rho_max/4 + jnp.zeros(N_t) 

    rho = np.zeros((N_t, N_x))
    rho[0,:] = rho_0x
    rho[:,0] = rho_t0
    rho[:,-1] = rho_t1
    I = np.ones(N_x)*2
    I[70:72] = 0.5

    dx = (x_N - x_0)/(N_x-1)
    L = np.full(I.shape, dx)

    return rho, I, L