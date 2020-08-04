from daft import PGM, Node


def hierarchical_p():
    """A naive representation of the hierarchical p that we desire."""
    G = PGM()
    G.add_node("p_shop", content=r"$p_{j, i}$", x=1, y=2, scale=1.2)
    G.add_node(
        "likes", content="$l_{j, i}$", x=1, y=1, scale=1.2, observed=True
    )
    G.add_node("p_owner", content=r"$p_{j}$", x=1, y=3, scale=1.2)
    G.add_node("p_pop", content=r"$p$", x=1, y=4, scale=1.2)

    G.add_edge("p_pop", "p_owner")
    G.add_edge("p_owner", "p_shop")
    G.add_edge("p_shop", "likes")

    G.add_plate(plate=[0.3, 0.3, 1.5, 2.2], label=r"shop $i$")
    G.add_plate(plate=[0, -0.1, 2.1, 3.6], label=r"owner $j$")

    G.render()


def convoluted_hierarchical_p():
    G = PGM()
    G.add_node(
        "likes", content="$l_{j, i}$", x=1, y=1, scale=1.2, observed=True
    )
    G.add_node("p_shop", content="$p_{j, i}$", x=1, y=2, scale=1.2)
    G.add_node("alpha_owner", content=r"$\alpha_{j}$", x=0, y=3, scale=1.2)
    G.add_node("beta_owner", content=r"$\beta_{j}$", x=2, y=3, scale=1.2)
    G.add_node(
        "lambda_a_pop", content=r"$\lambda_{\alpha}$", x=0, y=4, scale=1.2
    )
    G.add_node(
        "lambda_b_pop", content=r"$\lambda_{\beta}$", x=2, y=4, scale=1.2
    )
    G.add_node(
        "tau_lambda_a",
        content=r"$\tau_{\lambda_{\alpha}}$",
        x=0,
        y=5,
        fixed=True,
    )
    G.add_node(
        "tau_lambda_b",
        content=r"$\tau_{\lambda_{\beta}}$",
        x=2,
        y=5,
        fixed=True,
    )

    G.add_edge("alpha_owner", "p_shop")
    G.add_edge("beta_owner", "p_shop")
    G.add_edge("p_shop", "likes")
    G.add_edge("lambda_a_pop", "alpha_owner")
    G.add_edge("lambda_b_pop", "beta_owner")
    G.add_edge("tau_lambda_a", "lambda_a_pop")
    G.add_edge("tau_lambda_b", "lambda_b_pop")

    G.add_plate(plate=[0.5, 0.2, 1, 2.3], label=r"shop $i$")
    G.add_plate(plate=[-0.5, 0, 3, 3.5], label=r"owner $j$")
    G.render()


import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit
import numpy as np
import seaborn as sns


def plot_mu_p(mu, sigma):
    xs = np.linspace(mu - sigma * 4, mu + sigma * 4, 1000)
    ys = norm(loc=mu, scale=sigma).pdf(xs)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
    ax[0].plot(xs, ys)
    ax[0].set_xlabel(r"$\mu$")
    ax[0].set_ylabel("PDF")
    ax[0].axvline(x=0, color="red")
    ax[0].set_title("Gaussian Space")
    ax[1].plot(expit(xs), ys)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel(r"p = invlogit($\mu$)")
    ax[1].set_title("Bounded space")
    sns.despine()
    plt.show()


def hierarchical_pgm():
    G = PGM()

    tfm_plot_params = {"ec": "red"}

    G.add_node(
        "likes", content=r"$l_{j,i}$", x=0, y=0, scale=1.2, observed=True
    )
    G.add_node(
        "p_shop",
        content=r"$p_{j,i}$",
        x=0,
        y=1,
        scale=1.2,
        plot_params=tfm_plot_params,
    )
    G.add_node("mu_shop", content=r"$\mu_{j,i}$", x=1, y=1, scale=1.2)
    G.add_node("mu_owner", content=r"$\mu_{j}$", x=1, y=2, scale=1.2)
    G.add_node("sigma_owner", content=r"$\sigma_{j}$", x=2, y=2, scale=1.2)
    G.add_node(
        "p_owner",
        content=r"$p_{j}$",
        x=0,
        y=2,
        scale=1.2,
        plot_params=tfm_plot_params,
    )
    G.add_node("mu_population", content=r"$\mu$", x=1, y=3, scale=1.2)
    G.add_node(
        "sigma_population",
        content=r"$\sigma$",
        x=2,
        y=3,
        scale=1.2,
        fixed=True,
    )
    G.add_node(
        "p_population",
        content="p",
        x=0,
        y=3,
        scale=1.2,
        plot_params=tfm_plot_params,
    )
    G.add_node("lambda", content=r"$\lambda$", x=3, y=2, scale=1.2, fixed=True)
    G.add_node(
        "mean_population", content="mean", x=1, y=4, scale=1.2, fixed=True
    )
    G.add_node(
        "variance_population",
        content="variance",
        x=2,
        y=4,
        scale=1.2,
        fixed=True,
    )

    G.add_edge("mu_shop", "p_shop")
    G.add_edge("p_shop", "likes")
    G.add_edge("mu_owner", "mu_shop")
    G.add_edge("sigma_owner", "mu_shop")
    G.add_edge("mu_owner", "p_owner")
    G.add_edge("mu_population", "mu_owner")
    G.add_edge("sigma_population", "mu_owner")
    G.add_edge("mu_population", "p_population")
    G.add_edge("lambda", "sigma_owner")
    G.add_edge("mean_population", "mu_population")
    G.add_edge("variance_population", "mu_population")

    G.add_plate([-0.5, -0.5, 2, 2], label="shop $i$", position="bottom right")
    G.add_plate(
        [-0.7, -0.7, 3.2, 3.2], label="owner $j$", position="bottom right"
    )

    G.render()


import pymc3 as pm


def ice_cream_hierarchical_model(data):
    """Hierarchical model for ice cream shops"""
    n_owners = len(data["owner_idx"].unique())
    with pm.Model() as model:
        logit_p_overall = pm.Normal("logit_p_overall", mu=0, sigma=1)
        logit_p_owner_mean = pm.Normal(
            "logit_p_owner_mean",
            mu=logit_p_overall,
            sigma=1,
            shape=(n_owners,),
        )
        logit_p_owner_scale = pm.Exponential(
            "logit_p_owner_scale", lam=1 / 5.0, shape=(n_owners,)
        )
        logit_p_shop = pm.Normal(
            "logit_p_shop",
            mu=logit_p_owner_mean[data["owner_idx"]],
            sigma=logit_p_owner_scale[data["owner_idx"]],
            shape=(len(data),),
        )

        p_overall = pm.Deterministic("p_overall", pm.invlogit(logit_p_overall))
        p_shop = pm.Deterministic("p_shop", pm.invlogit(logit_p_shop))
        p_owner = pm.Deterministic("p_owner", pm.invlogit(logit_p_owner_mean))
        like = pm.Binomial(
            "like",
            n=data["num_customers"],
            p=p_shop,
            observed=data["num_favs"],
        )
    return model
