import numpy as np


def ECDF(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points
    n = len(data)

    # x-data for the ECDF
    x = np.sort(data)

    # y-data for the ECDF
    y = np.arange(1, n+1) / n

    return x, y


def despine(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
def despine_traceplot(traceplot):
    for row in traceplot:
        for ax in row:
            despine(ax)
