import matplotlib.pyplot as plt
from scipy.stats import beta, poisson, norm
import numpy as np


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
