import os
import requests
from dotenv import load_dotenv

from src.config.config_loader import load_config

# Load environment variables from .env
load_dotenv()

# Load project configuration
config = load_config()

# Get the API endpoint from config.yaml
url = config["data_sources"]["api_endpoints"]["json"]

# Get the application token from .env
app_token = os.getenv("NYC_OPEN_DATA_APP_TOKEN")

# Request headers
headers = {
    "X-App-Token": app_token,
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# Request body: fetch only 5 records
payload = {
    "query": "SELECT *",
    "page": {
        "pageNumber": 1,
        "pageSize": 5,
    },
    "includeSynthetic": False,
}

# Make the API request
response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=30,
)

# Check the response
print("HTTP Status:", response.status_code)

response.raise_for_status()

data = response.json()

print("Records received:", len(data))
print("Response type:", type(data).__name__)

if data:
    print("\nColumn names:")
    print(list(data[0].keys()))

    print("\nFirst record:")
    print(data[0])