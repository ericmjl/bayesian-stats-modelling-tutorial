[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ericmjl/bayesian-stats-modelling-tutorial/master)

# bayesian-stats-modelling-tutorial

How to do Bayesian statistical modelling using numpy and PyMC3

# getting started

To get started, first identify whether you:

1. Prefer to use the `conda` package manager (which ships with the Anaconda distribution of Python), or if you
2. prefer to use `pipenv`, which is a package authored by Kenneth Reitz for package management with `pip` and `virtualenv`, or if you
3. Do not want to mess around with dev-ops.
4. Only want to view the website version of the notebooks.

## 1. Clone the repository locally

In your terminal, use `git` to clone the repository locally.

```bash
git clone https://github.com/ericmjl/bayesian-stats-modelling-tutorial
```

Alternatively, you can download the zip file of the repository at the top of the main page of the repository. 
If you prefer not to use git or don't have experience with it, this a good option.

## 2. Download Anaconda (if you haven't already)

If you do not already have the [Anaconda distribution](https://www.anaconda.com/download/) of Python 3, 
go get it 
(note: you can also set up your project environment w/out Anaconda using `pip` to install the required packages; 
however Anaconda is great for Data Science and we encourage you to use it).

## 3. Set up your environment

### 3a. `conda` users

If this is the first time you're setting up your compute environment, 
use the `conda` package manager 
to **install all the necessary packages** 
from the provided `environment.yml` file.

```bash
conda env create -f binder/environment.yml
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
conda env update -f binder/environment.yml
```

### 3b. `pip` users

Please install all of the packages listed in the `environment.yml` file manually. 
An example command would be:

```bash
pip install networkx scipy ...
```

### 3c. don't want to mess with dev-ops

If you don't want to mess around with dev-ops, click the following badge to get a Binder session on which you can compute and write code.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ericmjl/bayesian-stats-modelling-tutorial/master)


### 4a. Open your Jupyter notebook

In the terminal, execute `jupyter notebook`.

Navigate to the notebooks directory 
and open the notebook `01-Student-Probability_a_simulated_introduction.ipynb`.

### 4b. Open your Jupyter notebook in Jupyter Lab!


In the terminal, execute `jupyter lab`.

Navigate to the notebooks directory 
and open the notebook `01-Student-Probability_a_simulated_introduction.ipynb`.

Now, if you're using Jupyter lab, for Notebook 2, you'll need to get ipywidgets working. 
The documentation is [here](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension).

In short, you'll need node installed & you'll need to run the following in your terminal:

`jupyter labextension install @jupyter-widgets/jupyterlab-manager`

### 4c. Open your Jupyter notebook using Binder.

Launch Binder using the button at the top of this README.md. Voila!

### 4d. Want to view static HTML notebooks

If you're interested in only viewing the static HTML versions of the notebooks, 
the links are provided below:

Part 1: Bayesian Data Science by Simulation

- [Introduction to Probability](https://ericmjl.github.io/bayesian-stats-modelling-tutorial/notebooks/01-Instructor-Probability_a_simulated_introduction.html)
- [Parameter Estimation and Hypothesis Testing](https://ericmjl.github.io/bayesian-stats-modelling-tutorial/notebooks/02-Instructor-Parameter_estimation_hypothesis_testing.html)

Part 2: Bayesian Data Science by Probabilistic Programming

- [Two Group Comparisons: Drug effect on IQ](https://ericmjl.github.io/bayesian-stats-modelling-tutorial/notebooks/03-instructor-two-group-iq.html)
- [Multi-Group Comparisons: Multiple ways of sterilizing phones](https://ericmjl.github.io/bayesian-stats-modelling-tutorial/notebooks/04-instructor-multi-group-comparsion-sterilization.html)
- [Two Group Comparisons: Darwin's Finches](https://ericmjl.github.io/bayesian-stats-modelling-tutorial/notebooks/05-instructor-two-group-comparison-finches.html)
- [Hierarchical Modelling: Baseball](https://ericmjl.github.io/bayesian-stats-modelling-tutorial/notebooks/06-instructor-hierarchical-baseball.html)
- [Hierarchical Modelling: Darwin's Finches](https://ericmjl.github.io/bayesian-stats-modelling-tutorial/notebooks/07-instructor-hierarchical-finches.html)
- [Bayesian Curve Regression: Identifying Radioactive Element](https://ericmjl.github.io/bayesian-stats-modelling-tutorial/notebooks/08-bayesian-curve-regression.html)


# Acknowledgements

Development of this type of material is almost always a result of years of discussions between members of a community. 
We'd like to thank the community and to mention several people who have played pivotal roles in our understanding the the material: 
Michael Betancourt, 
Justin Bois, 
Allen Downey, 
Chris Fonnesbeck, 
Jake VanderPlas. 
Also, Andrew Gelman rocks!


# Feedback

Please leave feedback for us [here](https://ericma1.typeform.com/to/j88n8P)! 
We'll use this information to help improve the teaching and delivery of the material.

# data credits

Please see individual notebooks for dataset attribution.

# Further Reading & Resources

Further reading resources that are not specifically tied to any notebooks.

- [Visualization in Bayesian workflow](https://arxiv.org/abs/1709.01449)
- [PyMC3 examples gallery](http://docs.pymc.io/examples.html)
- [Bayesian Analysis Recipes](https://github.com/ericmjl/bayesian-analysis-recipes)
- [Communicating uncertainty about facts, numbers and science](https://royalsocietypublishing.org/doi/full/10.1098/rsos.181870)