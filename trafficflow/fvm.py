import jax
import jax.numpy as jnp

from . import rho_C, V_0, l_eff, rho_max, T

@jax.jit
def Q_e(rho):
    return jnp.maximum(0, jnp.where(rho<=rho_C, V_0*rho, (1-rho*l_eff)/T))

@jax.jit
def fvm_graph(rho, A, I, L, dt):
    n = A.shape[0]
    flux = jnp.where(jnp.isclose(I, 0), 0, I*Q_e(rho/I))
    f_0 = jnp.sum(flux, axis=0)
    f_1 = jnp.sum(flux, axis=1)
    mask = jnp.logical_not(jnp.isclose(jnp.outer(jnp.sum(A, axis=0), jnp.sum(A, axis=1))*A,0))
    u = jnp.outer(f_0, jnp.ones(n))*mask
    w = jnp.outer(jnp.ones(n), f_1)*mask
    v = flux*mask

    rho_0 = jnp.sum(rho, axis=0)
    rho_1 = jnp.sum(rho, axis=1)
    u_local = jnp.outer(rho_0, jnp.ones(n))
    w_local = jnp.outer(jnp.ones(n), rho_1)
    v_local = rho

    d_inv = jnp.pow(jnp.sum(A, axis=1), -1)
    D_inv = jnp.diag(jnp.where(jnp.isinf(d_inv), 0, d_inv))
    u = jnp.dot(D_inv, u)
    u_local = jnp.dot(D_inv, u_local)

    mi1 = jnp.minimum(u, v)
    ma1 = jnp.maximum(u, v)
    Q_up = jnp.where(u_local<=v_local, mi1, ma1)

    mi2 = jnp.minimum(v, w)
    ma2 = jnp.maximum(v, w)
    Q_down = jnp.where(v_local<=w_local, mi2, ma2)

    l_inv = jnp.pow(L, -1)
    L_inv = jnp.where(jnp.isinf(l_inv), 0, l_inv)
    rho = rho - dt*L_inv*(Q_down - Q_up)
    return jnp.clip(rho, 0, rho_max)


@jax.jit
def g_godunov(u, v, I_u, I_v): 
    mi = jnp.minimum(I_u*Q_e(u/I_u), I_v*Q_e(v/I_v))
    ma = jnp.maximum(I_u*Q_e(u/I_u), I_v*Q_e(v/I_v))
    return jnp.where(u<=v, mi, ma)


@jax.jit
def fvm_2d(rho, I, L, dt):
    f_left = g_godunov(rho[:-2], rho[1:-1], I[:-2], I[1:-1])
    f_right = g_godunov(rho[1:-1], rho[2:], I[1:-1], I[2:])

    l_inv = jnp.pow(L[1:-1], -1)
    L_inv = jnp.where(jnp.isinf(l_inv), 0, l_inv)
    
    return rho[1:-1] - dt*L_inv*(f_right - f_left)
 