from scipy.stats import bernoulli, binom, multinomial
import numpy as np


def coin_distribution():
    return {
        "H": 1 / 3,
        "T": 2 / 3,
    }


def test_coin_distribution():
    cred_points = coin_distribution().values()
    assert np.allclose(sum(cred_points), 1)


def dice_distribution():
    return {
        1: 1 / 6,
        2: 1 / 6,
        3: 1 / 6,
        4: 1 / 6,
        5: 1 / 6,
        6: 1 / 6,
    }


def test_dice_distribution():
    cred_points = dice_distribution().values()
    assert np.allclose(sum(cred_points), 1)


def binomial_answer():
    binomial_dist = binom(n=1, p=0.8)
    draws = binomial_dist.rvs(10)
    return draws


def multinomial_answer():
    multinomial_dist = multinomial(n=1, p=[1 / 6] * 6)
    draws = multinomial_dist.rvs(10)
    return draws


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


def spaces_of_p():
    answer = """
There are actually infinite number of possible Bernoullis that we could instantiate!

Because there are an infinite set of numbers in the [0, 1] interval,
therefore there are an infinite number of possible Bernoullis that we could instantiate.
"""
    return answer


def spaces_of_data():
    answer = """
If you assume that there are no restrictions on the outcome,
then there should be $2^5$ ways to configure five Bernoulli draws.

More generally...

First off, there's no reason why we always have to have three 1s and two 0s in five draws;
it could have been five 1s or five 0s.
Secondly, the order of data (though it doesn't really matter in this case)
for three 1s and two 0s might well have been different.
"""
    return answer
