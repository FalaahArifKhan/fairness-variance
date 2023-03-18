import os
import pathlib
from dotenv import load_dotenv
from pymongo import MongoClient


def connect_to_mongodb():
    secrets_path = pathlib.Path(__file__).parent.joinpath('..', 'configs', 'secrets.env')
    load_dotenv(secrets_path)  # Take environment variables from .env

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    connection_string = os.getenv("CONNECTION_STRING")
    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(connection_string)
    collection = client[os.getenv("DB_NAME")]['preprocessing_results']

    def db_writer_func(run_models_metrics_df, collection=collection):
        # Rename Pandas columns to lower case
        run_models_metrics_df.columns = run_models_metrics_df.columns.str.lower()
        collection.insert_many(run_models_metrics_df.to_dict('records'))

    return client, collection, db_writer_func
