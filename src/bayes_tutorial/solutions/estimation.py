import janitor
import numpy as np
import pandas as pd
from pyprojroot import here
import pymc3 as pm
import arviz as az


def naive_estimate(data):
    estimated_p = data.join_apply(
        lambda x: x["num_favs"] / x["num_customers"]
        if x["num_customers"] > 0
        else np.nan,
        "p_hat",
    )
    return estimated_p


def assumptions():
    ans = """
When we choose to represent p_hat with 0,
we are oftentimes implicitly placing a strong assumption
that the estimate 0 is an unbiased estimator of the true p.

When we choose to represent p_hat with 0,
we are implicitly placing a strong assumption
that we don't have enough information to know p.

Either way, we have assumed something,
and there's no "objectivity" escape hatch here.
"""
    return ans


def ice_cream_store_model(data: pd.DataFrame) -> pm.Model:
    with pm.Model() as model:
        p = pm.Beta("p", alpha=2, beta=2, shape=(len(data),))
        like = pm.Binomial(
            "like", n=data["num_customers"], p=p, observed=data["num_favs"]
        )
    return model


def posterior_quantile(trace, q):
    trace_reshaped = trace.posterior.stack(draws=("chain", "draw"))
    return trace_reshaped.quantile(q=q, dim="draws").to_dataframe()


def trace_all_stores(data):
    with ice_cream_store_model(data):
        trace = pm.sample(2000)
        trace = az.from_pymc3(trace, coords={"p_dim_0": data["shopname"]})
    return trace


from daft import PGM


def ice_cream_one_group_pgm():
    G = PGM()
    G.add_node("alpha", content=r"$\alpha$", x=-1, y=1, scale=1.2, fixed=True)
    G.add_node("beta", content=r"$\beta$", x=1, y=1, scale=1.2, fixed=True)

    G.add_node("p", content="p", x=0, y=1, scale=1.2)
    G.add_node("likes", content="l", x=0, y=0, scale=1.2, observed=True)
    G.add_edge("alpha", "p")
    G.add_edge("beta", "p")
    G.add_edge("p", "likes")
    G.show()


def ice_cream_n_group_pgm():
    G = PGM()
    G.add_node("alpha", content=r"$\alpha$", x=-1, y=1, scale=1.2, fixed=True)
    G.add_node("beta", content=r"$\beta$", x=1, y=1, scale=1.2, fixed=True)

    G.add_node("p", content=r"$p_{i}$", x=0, y=1, scale=1.2)
    G.add_node("likes", content=r"$l_{i}$", x=0, y=0, scale=1.2, observed=True)
    G.add_edge("alpha", "p")
    G.add_edge("beta", "p")
    G.add_edge("p", "likes")
    G.add_plate([-0.5, -0.8, 1, 2.3], label=r"shop $i$")
    G.show()
