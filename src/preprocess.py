import pandas as pd

def load_data(path="data/students.csv"):
    df = pd.read_csv(path)
    return df

def split_features_target(df):

    X = df.drop(columns=["passed"])
    y = df["passed"]

    return X, y