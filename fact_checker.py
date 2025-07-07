import requests

def query_google_fact_check(claim):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": claim,
        "key": "YOUR_GOOGLE_FACT_CHECK_API_KEY"
    }
    res = requests.get(url, params=params)
    data = res.json()
    return data.get("claims", [])
