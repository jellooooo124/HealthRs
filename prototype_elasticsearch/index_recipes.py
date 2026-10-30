"""Index existing HealthRec processed catalog, or the repo's bundled Food.com sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.foodcom import load_recipes
from src.data.health import label_recipes
from .settings import es_client
from .schema import to_document, MAPPING

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = "healthrec-recipes-v1"


def read_catalog(source: str) -> tuple[pd.DataFrame, dict]:
    rules = yaml.safe_load((ROOT / "configs/health_rules.yaml").read_text(encoding="utf-8"))
    if source == "processed":
        path = ROOT / "data/processed/recipes.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing. Run python scripts/prepare_data.py in original HealthRec.")
        df = pd.read_parquet(path)
    elif source == "sample":
        df = load_recipes(str(ROOT / "data/sample"))
    else:
        raise ValueError(f"Unknown source: {source}")
    if (any("ok_" + name not in df.columns for name in rules["conditions"])
            or any("contains_" + name not in df.columns for name in rules["allergens"])):
        df = df.join(label_recipes(df, rules))
    return df, rules


def index_catalog(source: str = "sample", index: str = DEFAULT_INDEX, recreate: bool = False,
                  limit: int | None = None) -> dict:
    from elasticsearch.helpers import streaming_bulk
    es = es_client()
    df, rules = read_catalog(source)
    if limit is not None:
        df = df.head(limit)
    if recreate and es.indices.exists(index=index):
        es.indices.delete(index=index)
    if not es.indices.exists(index=index):
        es.indices.create(index=index, **MAPPING)
    conds, algs = list(rules["conditions"]), list(rules["allergens"])

    def records():
        for row in df.to_dict("records"):
            doc = to_document(row, conds, algs)
            yield {"_index": index, "_id": str(doc["recipe_id"]), "_source": doc}

    ok_count, failures = 0, []
    for ok, info in streaming_bulk(es, records(), chunk_size=400, raise_on_error=False, max_retries=2):
        if ok:
            ok_count += 1
        elif len(failures) < 5:
            failures.append(info)
    es.indices.refresh(index=index)
    return {"index": index, "source": source, "indexed": ok_count, "failed_examples": failures,
            "total_rows": len(df), "count_on_es": es.count(index=index)["count"]}


def main():
    parser = argparse.ArgumentParser(description="Index HealthRec Food.com recipes in Elasticsearch")
    parser.add_argument("--source", choices=("sample", "processed"), default="sample")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--recreate", action="store_true", help="DANGER: delete existing index before indexing")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(index_catalog(source=args.source, index=args.index, recreate=args.recreate, limit=args.limit))


if __name__ == "__main__":
    main()
