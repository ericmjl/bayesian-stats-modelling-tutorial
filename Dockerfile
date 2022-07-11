# Base image: miniconda3
FROM continuumio/miniconda3

# Install GCC for Theano
RUN apt-get install build-essential -y

# Install environment
COPY ./environment.yml /environment.yml
RUN conda env create -f /environment.yml
RUN rm /environment.yml

ENV PATH /opt/conda/envs/bayesian-modelling-tutorial/bin:$PATH
# For debugging purposes during environment build
RUN conda list -n bayesian-modelling-tutorial

# Install jupyterlab extensions
RUN jupyter labextension install @pyviz/jupyterlab_pyviz
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Copy contents of repository to Dockerfile
COPY . /root/bayes
WORKDIR /root/bayes

# Create Jupyter kernel to match notebooks
RUN python -m ipykernel install --user --name bayesian-stats-modelling-tutorial

# Entry point is Jupyter lab
ENTRYPOINT jupyter lab --port 8999 --ip="*" --allow-root --no-browser
