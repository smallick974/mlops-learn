import os
from dotenv import load_dotenv

load_dotenv()

datastore_name = os.getenv('datastore_name')
container_name = os.getenv('container_name')
account_name = os.getenv('account_name')
account_key = os.getenv('account_key')

