import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np


def plot_betadist_pdf(a, b):
    b = beta(a, b)
    x = np.linspace(0, 1, 1000)
    pdf = b.pdf(x)
    plt.plot(x, pdf)
