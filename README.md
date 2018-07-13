[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ericmjl/bayesian-stats-modelling-tutorial/master)

# bayesian-stats-modelling-tutorial

How to do Bayesian statistical modelling using numpy and PyMC3

# getting started

To get started, first identify whether you:

1. Prefer to use the `conda` package manager (which ships with the Anaconda distribution of Python), or if you
2. prefer to use `pipenv`, which is a package authored by Kenneth Reitz for package management with `pip` and `virtualenv`, or if you
3. Do not want to mess around with dev-ops.

## 1. Clone the repository locally

In your terminal, use `git` to clone the repository locally.

```bash
git clone https://github.com/ericmjl/bayesian-stats-modelling-tutorial
```

Alternatively, you can download the zip file of the repository at the top of the main page of the repository. If you prefer not to use git or don't have experience with it, this a good option.

## 2. Download Anaconda (if you haven't already)

If you do not already have the [Anaconda distribution](https://www.anaconda.com/download/) of Python 3, go get it (note: you can also set up your project environment w/out Anaconda using `pip` to install the required packages; however Anaconda is great for Data Science and we encourage you to use it).

## 3. Set up your environment

### 3a. `conda` users

If this is the first time you're setting up your compute environment, use the `conda` package manager to **install all the necessary packages** from the provided `environment.yml` file.

```bash
conda env create -f environment.yml
```

To **activate the environment**, use the `conda activate` command.

```bash
conda activate bayesian-modelling-tutorial
```

**If you get an error activating the environment**, use the older `source activate` command.

```bash
source activate bayesian-modelling-tutorial
```

To **update the environment** based on the `environment.yml` specification file, use the `conda update` command.

```bash
conda env update -f environment.yml
```

### 3b. `pip` users

Please install all of the packages listed in the `environment.yml` file manually. An example command would be:

```bash
pip install networkx scipy ...
```

### 3c. don't want to mess with dev-ops

If you don't want to mess around with dev-ops, click the following badge to get a Binder session on which you can compute and write code.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ericmjl/bayesian-stats-modelling-tutorial/master)


# Acknowledgements

Development of this type of material is almost always a result of years of discussions between members of a community. We'd like to thank the community and to mention several people who have played pivotal roles in our understanding the the material: Michael Betancourt, Justin Bois, Allen Downey, Chris Fonnesbeck, Jake VanderPlas. Also, Andrew Gelman rocks!

# data credits

Please see individual notebooks for dataset attribution.

# Further Reading & Resources

Further reading resources that are not specifically tied to any notebooks.

- [Visualization in Bayesian workflow](https://arxiv.org/abs/1709.01449)
- [PyMC3 examples gallery](http://docs.pymc.io/examples.html)
- [Bayesian Analysis Recipes](https://github.com/ericmjl/bayesian-analysis-recipes)
