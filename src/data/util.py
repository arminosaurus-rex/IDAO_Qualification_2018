# -*- coding: utf-8 -*-


# -- ==loadDataset== --

from data.make_dataset import download_dataset, create_hdf
import pandas as pd

def loadDataset():
    """Load the data set and return it as a pandas data frame."""
    download_dataset()
    create_hdf()
    return pd.read_hdf('./data/processed/train.hdf', key='data')

# -- ==loadDataset== --


# -- ==trainTestSplit== --

def trainTestSplit(df):
    """
    Split the data set into a train and a test set.
    
    The last 7 days are used for testing. Returns two pandas data frames, input should usually be the output of `loadDataset`.
    """
    maxDay = df["date"].max()
    train = df[df["date"] <= maxDay - 7]
    test = df[df["date"] > maxDay - 7]
    return train, test

# -- ==trainTestSplit== --



# -- ==scoreResult== --

def scoreSubmission(predictor, verbose=False):
    """
    Executes and compute the score of a prediction function.
    
    Give a function the takes a training data frame as input and returns a prediction data frame as specified in the problem statement. Returns the score of the predictions without making further sanity checks. `verbose` can be set to true to print more output.
    """
    if verbose: print("loading data set...")
    train, test = trainTestSplit(loadDataset())
    if verbose: print("computing predictions...")
    predictions = predictor(train)
    if verbose: print("scoring answer...")
    return scoreResult(predictions, test, verbose=verbose)

# -- ==scoreResult== --