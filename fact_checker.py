import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not set in environment variables.")

def query_google_fact_check(claim):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": claim,
        "key": api_key
    }
    res = requests.get(url, params=params)
    data = res.json()
    return data.get("claims", [])
