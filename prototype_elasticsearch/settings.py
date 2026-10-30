"""Runtime connection settings for the self-hosted Elasticsearch demo."""
from __future__ import annotations
import os


def es_client():
    from elasticsearch import Elasticsearch
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")
    kwargs = {"hosts": [url], "request_timeout": 20, "max_retries": 2, "retry_on_timeout": True}
    if api_key:
        kwargs["api_key"] = api_key
    return Elasticsearch(**kwargs)
