
#!/usr/bin/env python
# coding: utf-8

import os
import sys

# Append the parent directory of the tests folder to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycode import batch

import pandas as pd
from datetime import datetime


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

    return df


S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
if S3_ENDPOINT_URL is not None:
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
else:
    options = None
    

"""Read data into DataFrame"""
df_input = test_read_data()
input_file = batch.get_input_path(2022, 1)
output_file = batch.get_output_path(2022, 1)

print(df_input)

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

# Actual Data
os.system('python pycode/batch.py 2022 1')
df_actual = pd.read_parquet(output_file, storage_options=options)

assert abs(df_actual['predicted_duration'].sum().round(2) - 31.51) < 0.1
