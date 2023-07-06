import os
import pathlib
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient


def connect_to_mongodb(collection_name, secrets_path: str = pathlib.Path(__file__).parent.joinpath('..', '..', 'configs', 'secrets.env')):
    load_dotenv(secrets_path)  # Take environment variables from .env

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    connection_string = os.getenv("CONNECTION_STRING")
    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(connection_string)
    collection = client[os.getenv("DB_NAME")][collection_name]

    def db_writer_func(run_models_metrics_df, collection=collection):
        # Rename Pandas columns to lower case
        run_models_metrics_df.columns = run_models_metrics_df.columns.str.lower()
        run_models_metrics_df = run_models_metrics_df.rename(columns={'model_seed': 'bootstrap_model_seed'})
        collection.insert_many(run_models_metrics_df.to_dict('records'))

    return client, collection, db_writer_func


def read_model_metric_dfs_from_db(collection, session_uuid):
    cursor = collection.find({'session_uuid': session_uuid, 'tag': 'OK'})
    records = []
    for record in cursor:
        del record['_id']
        records.append(record)

    model_metric_dfs = pd.DataFrame(records)

    # Capitalize column names to be consistent across the whole library
    new_column_names = []
    for col in model_metric_dfs.columns:
        new_col_name = '_'.join([c.capitalize() for c in col.split('_')])
        new_column_names.append(new_col_name)

    model_metric_dfs.columns = new_column_names
    return model_metric_dfs
