# bayesian-stats-modelling-tutorial
How to do Bayesian statistical modelling using numpy and PyMC3

# getting started

To get started, first identify whether you:

1. Prefer to use the `conda` package manager (which ships with the Anaconda distribution of Python), or if you
2. prefer to use `pipenv`, which is a package authored by Kenneth Reitz for package management with `pip` and `virtualenv`, or if you
3. Do not want to mess around with dev-ops.

## `conda` users

If this is the first time you're setting up your compute environment, use the `conda` package manager to **install all the necessary packages** from the provided `environment.yml` file.

```bash
conda env create -f environment.yml
```

To **activate the environment**, use the `conda activate` command.

```bash
conda activate bayesian-stats-modelling
```

**If you get an error activating the environment**, use the older `source activate` command.

```bash
source activate bayesian-stats-modelling
```

To **update the environment** based on the `environment.yml` specification file, use the `conda update` command.

```bash
conda env update -f environment.yml
```

## `pipenv` users

Instructions are coming.

## don't want to mess with dev-ops

If you don't want to mess around with dev-ops, click the following badge to get a Binder session on which you can compute and write code.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ericmjl/bayesian-stats-modelling-tutorial/master)

# data credits

- [Baseball dataset](http://www.seanlahman.com/baseball-archive/statistics/)
