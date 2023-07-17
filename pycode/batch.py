#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
from tqdm.auto import tqdm
tqdm._instances.clear()

import numpy as np
import pandas as pd


def prepare_data(df, categorical) -> pd.DataFrame:
    df["tpep_dropoff_datetime"] = pd.to_datetime(df.tpep_dropoff_datetime)
    df["tpep_pickup_datetime"] = pd.to_datetime(df.tpep_pickup_datetime)

    df["duration"] = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # df[categorical] = df[categorical].astype(str)
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def read_data(filename, categorical) -> pd.DataFrame:
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    """Read data into DataFrame"""
    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)

    return prepare_data(df, categorical)


def predict_duration(df: pd.DataFrame, categorical, dv, lr) -> np.ndarray:
    """Predict the duration using the trained model"""
    dicts  = df[categorical].to_dict(orient='records')
    X_val  = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    return y_pred

        
def save_results(df: pd.DataFrame, y_pred: np.ndarray, output_file: str) -> None:
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred    
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')   

    """Save the predicted results to a parquet file""" 
    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }

        df.to_parquet(output_file, engine='pyarrow', index=False, storage_options=options)
    else:
        os.makedirs('output', exist_ok=True)        
        df_result.to_parquet(        
            output_file,
            engine='pyarrow',
            # compression=None,
            index=False,
            storage_options=None
        )
    return None 


def get_input_path(year, month):
    # default_input_pattern = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    default_input_pattern  = f'./data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    # default_output_pattern = f's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    default_output_pattern = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet' 

    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

    
def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
  
    steps = ["Loading model", "Reading data", "Predict data"]
    with tqdm(total=len(steps), desc="Running steps", leave=True) as pbar:
        # Step 1: Loading model
        pbar.set_description(steps[0])
        with open('models/model.bin', 'rb') as f_in:
            dv, lr = pickle.load(f_in)
        pbar.update(1)

        # Step 2: Reading data
        pbar.set_description(steps[1])
        df = read_data(input_file, categorical)
        df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
        pbar.update(1)
        
        # Step 3: Predict data
        pbar.set_description(steps[2])
        y_pred = predict_duration(df, categorical, dv, lr)
        pbar.update(1)
        pbar.close()    
        

    # Print Prediction
    print('predicted mean duration:', y_pred.mean().round(2))
    print('predicted sum duration:', y_pred.sum().round(2))

    # save_results
    save_results(df, y_pred, output_file)

    
if __name__ == '__main__':    
    # Global Parameters
    year        = int(sys.argv[1]) # 2022
    month       = int(sys.argv[2]) # 2 
    
    main(year, month)
