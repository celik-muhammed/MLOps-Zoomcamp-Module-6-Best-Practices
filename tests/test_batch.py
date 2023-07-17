#!/usr/bin/env python
# coding: utf-8

import os
import sys

# Append the parent directory of the tests folder to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycode import batch
from datetime import datetime

import pickle
from tqdm.auto import tqdm
tqdm._instances.clear()

import numpy as np
import pandas as pd
from pandas import Timestamp
import pandas.testing as pd_testing


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_read_data() -> pd.DataFrame:
    """Read data into DataFrame"""
    data    = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]
    categorical = ['PULocationID', 'DOLocationID']   
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

    df = pd.DataFrame(data, columns=columns)

    df["tpep_dropoff_datetime"] = pd.to_datetime(df.tpep_dropoff_datetime)
    df["tpep_pickup_datetime"] = pd.to_datetime(df.tpep_pickup_datetime)

    df["duration"] = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # df[categorical] = df[categorical].astype(str)
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    print(df)

    actual_features = df
    # print(df.to_dict())
    
    expected_data = {
        'PULocationID': {0: '-1', 1: '1', 2: '1'}, 
        'DOLocationID': {0: '-1', 1: '-1', 2: '2'}, 
        'tpep_pickup_datetime': {0: Timestamp('2022-01-01 01:02:00'), 1: Timestamp('2022-01-01 01:02:00'), 2: Timestamp('2022-01-01 02:02:00')}, 
        'tpep_dropoff_datetime': {0: Timestamp('2022-01-01 01:10:00'), 1: Timestamp('2022-01-01 01:10:00'), 2: Timestamp('2022-01-01 02:03:00')}, 
        'duration': {0: 8.0, 1: 8.0, 2: 1.0}
    }
    expected_features = pd.DataFrame(expected_data)

    assert actual_features.equals(expected_features)
    assert (actual_features == expected_features).all().all()
    # pd_testing.assert_frame_equal(actual_features, expected_features)

    return df

    
if __name__ == '__main__':
    test_read_data()
