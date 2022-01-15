import streamlit as st
from configparser import ConfigParser
from mongo_util import SetUpMongo
import os


def initialize_streamlit_and_mongodb():
    # initialize configuration file
    config = ConfigParser()
    config.read('config.ini')

    # create directory for resources and models
    if not os.path.isdir('./resources'):
        os.mkdir('./resources')

    # Handling initialization for Topic model
#     if not os.path.isdir(config['topic_cluster']['model_path']):
#         os.mkdir(config['topic_cluster']['model_path'], 0o777)

    # initialize MongoDB Cloud Connection
    db_name = config['mongo']['database_name']
    connection_string = config['mongo']['connection_string'].format('google123', db_name)
    mongo_connect = SetUpMongo(db_name, connection_string)

    return mongo_connect, config
