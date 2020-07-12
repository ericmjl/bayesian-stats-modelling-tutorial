from scipy.stats import bernoulli
import numpy as np


def likelihood_coin_toss():
    answer = """
This does not surprise me.
Under a fair coin toss model,
a continuous sequence of 0s has the same likelihood
as a mixed sequence of 0s and 1s.
As such, fair coin tosses are expected to be quite "clumpy"!

HOWEVER...

A continuous sequence of 0s should compel me
to think about whether the fair coin toss model is right or not...
"""
    return answer


def fair_coin_model():
    return bernoulli(0.5)


def coin_data_likelihood(coin_data_1, coin_data_2):
    coin = fair_coin_model()
    return (coin.pmf(coin_data_1), coin.pmf(coin_data_2))


def coin_data_joint_likelihood(coin_data_1, coin_data_2):
    coin = fair_coin_model()
    return (
        np.product(coin.pmf(coin_data_1)),
        np.product(coin.pmf(coin_data_2)),
    )


def coin_data_joint_loglikelihood(coin_data_1, coin_data_2):
    coin = fair_coin_model()
    return (np.sum(coin.logpmf(coin_data_1)), np.sum(coin.logpmf(coin_data_2)))
