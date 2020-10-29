import matplotlib.pyplot as plt
from scipy.stats import beta, poisson, norm
import numpy as np
import pymc3 as pm
import arviz as az
from functools import lru_cache


def plot_betadist_pdf(a, b):
    b = beta(a, b)
    x = np.linspace(0, 1, 1000)
    pdf = b.pdf(x)
    plt.plot(x, pdf)


def coin_flip_data():
    return np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1,])


def car_crash_data():
    return poisson(3).rvs(25)


def finch_beak_data():
    return norm(12, 1).rvs(38)


def car_crash_model_generator():
    data = car_crash_data()
    with pm.Model() as car_crash_model:
        mu = pm.Exponential("mu", lam=1 / 29.0)
        like = pm.Poisson("like", mu=mu, observed=data)
    return car_crash_model


def car_crash_interpretation():
    ans = """
We believe that the rate of car crashes per week
is anywhere from between 2.3 to 3.6 (94% posterior HDI),
with a mean of 2.9.
"""
    return ans


def model_inference_answer(model):
    with model:
        trace = pm.sample(2000)
        trace = az.from_pymc3(trace)
    return trace


def model_trace_answer(trace):
    az.plot_trace(trace)


def model_posterior_answer(trace):
    az.plot_posterior(trace)


def finch_beak_model_generator():
    data = finch_beak_data()
    with pm.Model() as finch_beak_model:
        mu = pm.Normal("mu", mu=10, sigma=3)
        sigma = pm.Exponential("sigma", lam=1 / 29.0)
        like = pm.Normal("like", mu=mu, sigma=sigma, observed=data)
    return finch_beak_model


def finch_beak_interpretation():
    ans = """
Having seen the data, we believe that finch beak lengths
are expected to be between 11.7 and 12.1 (approx, 94% HDI),
with an average of 11.9.

We also believe that the intrinsic variance of finch beak lengths
across the entire population of finches
is estimated to be around 0.56 to 0.87,
with an average of 0.7.
"""
    return ans
