import pandas as pd
from sklearn.preprocessing import LabelEncoder
import janitor as jn
import numpy as np


def load_finches_2012():
    path = '../data/finch_beaks_2012.csv'
    return load_finches(path)


def load_finches_1975():
    path = '../data/finch_beaks_1975.csv'
    df = load_finches(path)
    df = df.rename_column('beak_length_mm', 'beak_length').rename_column('beak_depth_mm', 'beak_depth')
    return df


def load_finches(path):
    # Load the data
    df = (
        pd.read_csv(path)
        .clean_names()    # clean column names
        .rename_column('blength', 'beak_length')  # rename blength to beak_length (readability fix)
        .rename_column('bdepth', 'beak_depth')   # rename bdepth to beak_depth (readability fix)
        .label_encode('species')  # create a `species_enc` column that has the species encoded numerically
    )
    return df


def load_baseball():
    df = pd.read_csv('../data/baseballdb/core/Batting.csv')
    df['AB'] = df['AB'].replace(0, np.nan)
    df = df.dropna()
    df['batting_avg'] = df['H'] / df['AB']
    df = df[df['yearID'] >= 2016]
    df = df.iloc[0:15]
    df.head(5)
    return df


def load_sterilization():
    df = (
        pd.read_csv('../data/sterilization.csv', na_filter=True, na_values=['#DIV/0!'])
        .clean_names()
        .label_encode('treatment')
    )
    mapping = dict(zip(df['treatment'], df['treatment_enc']))
    return df, mapping


def load_kruschke():
    df = (
        pd.read_csv('../data/iq.csv', index_col=0)  # comment out the path to the file for students.
        .label_encode('treatment')
    )
    return df


# Constants for load_decay
tau = 71.9 # indium decay half life
A = 42  # starting magnitude
C = 21 # measurement error
noise_scale = 1


def load_decay():
    t = np.arange(0, 1000)
    def decay_func(ts, noise):
        return A  * np.exp(-t/tau) + C + np.random.normal(0, noise, size=(len(t)))

    data = {'t': t, 'activity': decay_func(t, noise_scale)}
    df = pd.DataFrame(data)
    return df