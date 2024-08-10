import osmnx as ox
import matplotlib.pyplot as plt

from trafficflow.generate import init_graph_2d
from trafficflow.fvm import fvm_2d
from trafficflow import rho_max
from trafficflow.utils import plot_osm_graph


def main():
    x_0, x_N = 0, 100
    N_x = 101
    T_sim = 5
    N_t = 5000

    dx = (x_N - x_0)/(N_x-1)
    dt = T_sim/N_t


    rho, I, L = init_graph_2d(rho_max, N_x, N_t, x_0, x_N)

    for t_i in range(N_t-1):
        output = fvm_2d(rho[t_i], I, L, dt)
        rho[t_i+1, 1:-1] = output

    f, ax = plt.subplots(1, figsize=(10,3))
    pos = ax.imshow(rho, aspect='auto', interpolation='none', vmin=0, vmax=120)
    f.colorbar(pos, ax=ax)
    plt.tight_layout()

    f.savefig('images/fvm_2d.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()