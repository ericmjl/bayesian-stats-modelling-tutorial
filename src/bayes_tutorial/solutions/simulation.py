import matplotlib.pyplot as plt
from typing import List
import numpy as np
import scipy.stats as sts
from daft import PGM


def coin_flip_pgm():
    G = PGM()
    G.add_node("alpha", content=r"$\alpha$", x=-1, y=1, scale=1.2, fixed=True)
    G.add_node("beta", content=r"$\beta$", x=1, y=1, scale=1.2, fixed=True)
    G.add_node("p", content="p", x=0, y=1, scale=1.2)
    G.add_node("result", content="result", x=0, y=0, scale=1.2, observed=True)
    G.add_edge("alpha", "p")
    G.add_edge("beta", "p")
    G.add_edge("p", "result")
    G.show()


def coin_flip_generator_v2(alpha: float, beta: float) -> np.ndarray:
    """
    Coin flip generator for a `p` that is not precisely known.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be positive, but you passed in {alpha}")
    if beta < 0:
        raise ValueError(f"beta must be positive, but you passed in {beta}.")
    p = sts.beta(a=alpha, b=beta).rvs(1)
    result = sts.bernoulli(p=p).rvs(1)
    return result


def generate_many_coin_flips(
    n_draws: int, alpha: float, beta: float
) -> List[int]:
    """
    Generate n draws from the coin flip generator.
    """
    data = [coin_flip_generator_v2(alpha, beta) for _ in range(n_draws)]
    return np.array(data).flatten()


def coin_flip_joint_loglike(data: List[int], p: float) -> float:
    p_loglike = sts.beta(a=10, b=10).logpdf(
        p
    )  # evaluate guesses of `p` against the prior distribution
    data_loglike = sts.bernoulli(p=p).logpmf(data)

    return np.sum(data_loglike) + np.sum(p_loglike)


def car_crash_pgm():
    G = PGM()
    G.add_node("crashes", content="crashes", x=0, y=0, scale=1.5)
    G.add_node("rate", content="rate", x=0, y=1, scale=1.5)
    G.add_edge("rate", "crashes")
    G.show()


def car_crash_data():
    data = [1, 5, 2, 3, 8, 4, 5]
    return data


def car_crash_loglike(rate: float, crashes: List[int]) -> float:
    """Evaluate likelihood of per-week car crash data points."""
    rate_like = np.sum(sts.expon(scale=0.5).logpdf(rate))
    crashes_like = np.sum(sts.poisson(mu=rate).logpmf(crashes))
    return rate_like + crashes_like


def car_crash_loglike_plot():
    data = car_crash_data()
    _, ax = plt.subplots()
    rates = np.arange(0, 10, 0.1)
    loglike = []
    for rate in rates:
        loglike.append(car_crash_loglike(rate, data))
    ax.plot(rates, loglike)
    return ax


def korea_pgm():
    G = PGM()
    G.add_node("s_mean", r"$\mu_{s}$", x=0, y=1)
    G.add_node("s_scale", r"$\sigma_{s}$", x=1, y=1)
    G.add_node("s_height", r"$h_s$", x=0.5, y=0)
    G.add_edge("s_mean", "s_height")
    G.add_edge("s_scale", "s_height")

    G.add_node("n_mean", r"$\mu_{n}$", x=2, y=1)
    G.add_node("n_scale", r"$\sigma_{n}$", x=3, y=1)
    G.add_node("n_height", r"$h_n$", x=2.5, y=0)
    G.add_edge("n_mean", "n_height")
    G.add_edge("n_scale", "n_height")

    G.show()


def s_korea_generator():
    s_korea_mean = sts.norm(loc=180, scale=3).rvs()
    s_korea_scale = sts.expon(scale=1).rvs()
    height = sts.norm(loc=s_korea_mean, scale=s_korea_scale).rvs()
    return height


def n_korea_generator():
    n_korea_mean = sts.norm(loc=165, scale=3).rvs()
    n_korea_scale = sts.expon(scale=1).rvs()
    height = sts.norm(loc=n_korea_mean, scale=n_korea_scale).rvs()
    return height


def s_korea_height_loglike(
    mean: float, scale: float, heights: List[float]
) -> float:
    mean_loglike = sts.norm(loc=180, scale=3).logpdf(mean)
    scale_loglike = sts.expon(scale=1).logpdf(scale)
    height_loglike = sts.norm(loc=mean, scale=scale).logpdf(heights)
    return (
        np.sum(height_loglike) + np.sum(mean_loglike) + np.sum(scale_loglike)
    )


def n_korea_height_loglike(
    mean: float, scale: float, heights: List[float]
) -> float:
    mean_loglike = sts.norm(loc=165, scale=3).logpdf(mean)
    scale_loglike = sts.expon(scale=1).logpdf(scale)
    height_loglike = sts.norm(loc=mean, scale=scale).logpdf(heights)
    return (
        np.sum(height_loglike) + np.sum(scale_loglike) + np.sum(mean_loglike)
    )


def joint_height_loglike(
    s_mean: float,
    s_scale: float,
    n_mean: float,
    n_scale: float,
    s_heights: List[int],
    n_heights: List[int],
) -> float:
    s_korea_loglike = s_korea_height_loglike(s_mean, s_scale, s_heights)
    n_korea_loglike = n_korea_height_loglike(n_mean, n_scale, n_heights)
    return s_korea_loglike + n_korea_loglike


def korea_height_generator(
    mean_loc: float, mean_scale: float, scale_loc: float
) -> float:
    mean = sts.norm(loc=mean_loc, scale=mean_scale).rvs()
    scale = sts.expon(loc=scale_loc).rvs()
    height = sts.norm(loc=mean, scale=scale).rvs()
    return height


def s_korea_height_data():
    return [korea_height_generator(175, 0.3, 5) for _ in range(1000)]


def n_korea_height_data():
    return [korea_height_generator(165, 0.2, 3) for _ in range(1000)]
