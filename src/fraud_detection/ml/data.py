
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from fraud_detection.config import (COLUMNS_TO_CLEAN, DATA_CSV_PATH,
                                    DATA_PICKLE_PATH)


# Save data to pickle
def save_pickle_dataset(data: DataFrame, path: Path| str = DATA_PICKLE_PATH):
    """Save dataframe to pickle file

    Args:
        data (DataFrame): dataframe to save
        path (Path | str, optional): file pathname Defaults to DATA_PICKLE_PATH.
    """
    with open(path, 'wb') as _:
        data.to_pickle(path, compression='infer')


# Load data from pickle file
def load_pickle_dataset(path: Path| str= DATA_PICKLE_PATH) -> DataFrame:
    """Load pickle file into pandas dataframe

    Args:
        path (Path | str, optional): pickle file path. Defaults to DATA_PICKLE_PATH.

    Returns:
        DataFrame: dataframe
    """
    with open(path, 'rb') as output:
        return pd.read_pickle(output)


def clean_data(df: DataFrame) -> DataFrame:
    """
    Clean raw data by
    - removing duplicated column
    - removing unwanted columns
    """
    print("⏳ cleaning data...")
    df = df.drop_duplicates()
    df = df.drop(COLUMNS_TO_CLEAN, axis=1)
    print("✅ data cleaned\n")
    
    return df
    
    
def load_data() -> DataFrame:
    """Load data from pickle or csv
    """
    if Path(DATA_PICKLE_PATH).exists():
        print("⏳ Loading dataset from pickle...")
        data = load_pickle_dataset()
        print("✅ Data loaded\n")
    else:
        print("⏳ Loading dataset from csv...")
        data = pd.read_csv(f"{DATA_CSV_PATH}")
        print("✅ Data loaded \n")
        
        print("⏳ Saving dataset to pickle file...")
        save_pickle_dataset(data)
        print("✅ Data Saved n")
        
    return data