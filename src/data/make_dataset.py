import os.path
import pandas as pd
import pathlib
import urllib.request

def download_dataset():
    if not os.path.isfile("./data/raw/train.csv.zip"):
        # create directory
        pathlib.Path("./data/raw/").mkdir(parents=True, exist_ok=True)

        # download the file
        url = "https://www.dropbox.com/s/v71xw29hqt4qykb/train.csv.zip?dl=1"
        urllib.request.urlretrieve(url, "./data/raw/train.csv.zip")

def create_hdf():
    if not os.path.isfile("./data/processed/train.hdf"):
        # create directory
        pathlib.Path("./data/processed/").mkdir(parents=True, exist_ok=True)

        # convert data set to hdf
        training_data = pd.read_csv("./data/raw/train.csv.zip")
        training_data.to_hdf("./data/processed/train.hdf", key="data", complevel=1, complib="zlib")

if __name__ == "__main__":
    print("downloading data...")
    download_dataset()
    print("creating hdf...")
    create_hdf()
