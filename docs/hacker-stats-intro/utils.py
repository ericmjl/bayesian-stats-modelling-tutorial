import numpy as np


def ECDF(data):
    x = np.sort(data)
    y = np.cumsum(x) / np.sum(x)
    
    return x, y


def despine(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
def despine_traceplot(traceplot):
    for row in traceplot:
        for ax in row:
            despine(ax)