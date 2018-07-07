import pandas as pd
from sklearn.preprocessing import LabelEncoder
import janitor as jn
import numpy as np


def load_finches_2012():
    path = '../data/finch_beaks_2012.csv'
    return load_finches(path)

def load_finches_1975():
    path = '../data/finch_beaks_1975.csv'
    return load_finches(path)


def load_finches(path):
    # Load the data
    df = pd.read_csv(path)

    # Data cleaning methods. This is provided for you. Follow along the annotations
    # to learn what's going on.
    df = (jn.DataFrame(df)  # wrap dataframe in a Janitor dataframe.
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
    df = pd.read_csv('../data/sterilization.csv', na_filter=True, na_values=['#DIV/0!'])
    df = jn.DataFrame(df).clean_names().label_encode('treatment')
    mapping = dict(zip(df['treatment'], df['treatment_enc']))
    return df, mapping