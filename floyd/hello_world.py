import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="data file to use", default="/data/train.hdf")
    args = parser.parse_args()

    print("Hello world")
    print(args.data)

    dataframe = pd.read_hdf(args.data, key="data")

    print(dataframe)

