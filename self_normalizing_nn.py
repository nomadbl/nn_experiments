import torch.nn as nn
import sympy as sp
import sympy.stats as sp_stats

def self_normalizing_nn_init(layer: nn.Linear):
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity="linear")
    if not layer.bias is None:
        nn.init.constant_(layer.bias, 0)
    return layer

def selu(x, l, a):
    return l*sp.Piecewise((x, x > 0), (a*sp.exp(x)-a, x <= 0))

if __name__ == "__main__":
    # derive self normalizing nns using sympy
    mu = sp.Symbol("mu", positive=True)
    nu = sp.Symbol("nu", positive=True)
    o = sp.Symbol("o", positive=True)
    t = sp.Symbol("t", positive=True)
    l = sp.Symbol("l", positive=True)
    a = sp.Symbol("a", positive=True)
    w = sp_stats.Normal("w", o, sp.sqrt(t))
    x = sp_stats.Normal("x", mu, sp.sqrt(nu))
    # z = w * x
    z = x
    print(f"{z.subs(nu, 1).subs(t, 1).subs(o, 0)}")
    print(f"{sp_stats.density(z.subs(nu, 1).subs(t, 1).subs(o, 0))}")
    e = selu(z, l, a)
    e = e + x
    e = e.subs(mu, 0).subs(nu, 1).subs(t, 1).subs(o, 0)
    # mu_new = sp_stats.E(e)
    mu_new = sp_stats.E(e.subs(l,1).subs(a,1))
    # nu_new = sp_stats.variance(e)
    print(f"expectation value of activations is {mu_new}")
    # print(f"activation expectation value is {mu_new}")
    # print(f"activation expectation value is {nu_new}")
    # mu_new_of_interest = mu_new.subs(mu, 0).subs(nu, 1).subs(t, 1).subs(o, 0)
    # nu_new_of_interest = nu_new.subs(mu, 0).subs(nu, 1).subs(t, 1).subs(o, 0)
    # print(f"for mu=0, nu=1, t=1, o=0, this is mu_new={mu_new}")
    # print(f"for mu=0, nu=1, t=1, o=0, this is nu_new={nu_new}")

