# HealthRec 2.0 — Elasticsearch + Personalized Recommendation Prototype

**An additive extension to [jellooooo124/HealthRs](https://github.com/jellooooo124/HealthRs).** It preserves the existing nutrition/allergy labels, SQLite `ProfileStore`, and optionally the trained BPR-MF + Health-aware model. It adds an Elasticsearch lexical retrieval channel and an explainable, feedback-sensitive ranking path. It is not a clinical decision support product.

## What changes from original HealthRec?

| Original HealthRec | Elasticsearch prototype |
| --- | --- |
| Taste BPR top-N + healthy popularity candidate retrieval | Elasticsearch multi-field **BM25**, searchable names/descriptions/ingredients/keywords and structured range filters |
| `TwoStageRecommender.rank` on original candidates | Reuses original model **on Elasticsearch candidates**, when processed catalog/checkpoints and warm user are available |
| Profile/allergen and health filters | Mandatory **Elasticsearch filters**, followed by a Python fail-closed check before display |
| User's taste preferences in conversational agent | Explicit taste-term matching in transparent hybrid ranking |
| Like/dislike feedback kept in SQLite | Latest feedback additionally excludes disliked recipes and boosts the category of liked/cooked recipes |
| Streamlit chat with external LLM | Separate reproducible search/recommendation UI + JSON API requiring **no LLM key** |

### System flow

```
                           trusted ProfileStore (SQLite)
user text + constraints ----> conditions, allergies, taste, feedback
         |                                 |
         v                                 v
Elasticsearch BM25 + enforced query filters (100 candidates)
         |
         +--> OPTIONAL original HealthRec trained BPR-MF / Adaptation A
         |                                 |
         +----------> score fusion <--------+
                     BM25 + BPR + taste + popularity
                                 |
                     Python safety double-check
                                 |
                     category diversity -> top K
                                 |
                      scores, explanations, feedback
```

### Rank formula

For each candidate, component scores are **min–max normalized in the current candidate set**:

- **Warm, model available:** `0.25·BM25 + 0.40·BPR + 0.25·taste + 0.10·popularity`.
- **Cold start or unavailable model:** `0.40·BM25 + 0.40·taste + 0.20·popularity`.

Taste uses explicit taste preferences, and a same-category boost based on positive feedback. These are initial **heuristic weights**, not learned weights or measured improvements. The response includes scores and weights for auditing. The original trained ranker is used only when the processed catalog and its serving checkpoints are available; otherwise `bpr_used: false` is explicit.

## Installation: inside the existing HealthRs repo

Place this `prototype_elasticsearch/` folder at the root of your original `HealthRs` checkout (next to `src/`, `agent/`, `configs/`, `data/`). From that root:

```bash
# Python 3.12; use the already configured HealthRec environment.
pip install -r requirements.txt
pip install -r prototype_elasticsearch/requirements-es.txt

docker compose -f prototype_elasticsearch/compose.yaml up -d
curl -s http://localhost:9200

# Index the bundled Food.com sample; do not need Kaggle or API key.
python -m prototype_elasticsearch.index_recipes --source sample --recreate

# API (terminal 1)
uvicorn prototype_elasticsearch.server:app --host 127.0.0.1 --port 8000

# UI (terminal 2)
streamlit run prototype_elasticsearch/ui.py
```

UI: `http://localhost:8501`. Interactive API docs: `http://localhost:8000/docs`.

The default index is `healthrec-recipes-v1`. Set `ELASTICSEARCH_URL`, `ELASTICSEARCH_API_KEY`, `HEALTHREC_ES_INDEX`, and `HEALTHREC_USE_BPR` as needed. `HEALTHREC_PROFILE_DB` optionally overrides the SQLite file. Docker Compose binds ES only to `127.0.0.1`. **Security is disabled only for a LOCAL development cluster.** Never deploy that Compose file to a public network or use real patient data without adding authentication, authorisation, transport security and governance.

For real trained HealthRec results, first run the original `python scripts/prepare_data.py` and ensure its serving checkpoints exist. Then run:

```bash
python -m prototype_elasticsearch.index_recipes --source processed --recreate
```

`--recreate` deletes the named ES index before rebuilding. Omit it to update recipe documents by ID. Never run it against a shared production index.

## Three-minute demonstration

Create/confirm a synthetic user profile (safety fields must be stored first):

```bash
curl -X PUT http://127.0.0.1:8000/profiles/demo_alice \
  -H 'Content-Type: application/json' \
  -d '{"conditions":["diabetes"],"allergies":["peanut"],"taste_prefs":["chicken","soup"],"confirmed":true}'
```

Search healthy recipe candidates and personalize:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo_alice","query":"chicken dinner","k":5,"candidate_size":100,"constraints":{"max_minutes":45,"max_calories":550}}'
```

Record feedback, then repeat the same query:

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo_alice","recipe_id":12345,"signal":"dislike"}'
```

Replace `12345` with the ID of an **actual returned recipe**. Disliked recipe IDs are excluded in Elasticsearch and double-checked during rerank. For `"like"` or `"cooked"`, the categories of liked recipes provide a modest ranking boost to similar-category candidates. Repeating the query demonstrates the effect.

## API

| Method | Route | Function |
| --- | --- | --- |
| GET | `/health` | Verify Elasticsearch index and count |
| GET | `/profiles/{user_id}` | Read or initialize a synthetic user |
| PUT | `/profiles/{user_id}` | Confirm conditions, allergies and taste preferences |
| POST | `/recommendations` | Run BM25 candidate retrieval, safety filters and personalized rerank |
| POST | `/feedback` | Store like/dislike/cooked/clear in the original SQLite format |

`/recommendations` accepts `user_id`, `query`, `k`, `candidate_size`, and `constraints` (`max_minutes`, `max_calories`, `category`, `include_ingredients`, `exclude_ingredients`). **It intentionally does not accept conditions/allergies as query parameters**: the backend reads them from the trusted profile. If the profile is unconfirmed, it refuses to return recipes.

## Safety and scope

1. **Allergy and condition rules precede ranking.** They are never traded for taste or BM25 relevance. The Python layer repeats the checks and rejects incomplete nutrition/ingredients.
2. **Existing `configs/health_rules.yaml` is reused.** The thresholds are simplified per-serving labels, **not** medical clearance or a substitute for professional dietetic advice. Ingredient keyword matching is heuristic and cannot reliably detect trace allergens or cross-contamination.
3. **Unknown or missing data are not certified safe.** Incomplete recipe nutrition is excluded. The implementation does not attempt to infer missing fields using an LLM.
4. **Feedback is per-user in SQLite.** The prototype is unauthenticated and for synthetic data on a developer machine only; real use needs access control, encryption, consent, retention/deletion policies, testing and independent nutrition verification.
5. Elasticsearch's role here is **recipe search**, not guideline retrieval. The original Chroma/RAG guideline system can continue unchanged.

## Evaluation

The repository adds unit tests for query-level safety constraints, secondary Python safety checks, personalization, feedback, profile confirmation, and index document shape:

```bash
python -m pytest prototype_elasticsearch/tests -q
```

For a real online demonstration, check `GET /health` after indexing and inspect the JSON `score_components`, `score_weights` and `bpr_used`. A future systematic experiment should compare `BM25`, `BM25 + taste/feedback`, and `BM25 + BPR + taste/feedback` on fixed temporal user splits, using **NDCG@10, Recall@10, allergen violation rate, condition violation rate, catalog coverage, cold-start utility and P95 latency**. Do not claim superiority without this experiment. BM25 is not guaranteed to improve rating-based NDCG, especially when the query does not reflect the held-out target recipe.

## Known limitations

- Full-model performance depends on `data/processed/`, serving checkpoints, and the original trained HealthRec environment; the sample serves only for functional demos.
- The sample and processed catalogs need to be reindexed whenever the underlying rules or recipe data change.
- BM25 is a lexical retriever; the prototype does not include semantic vector retrieval, embeddings, learning-to-rank, or feedback-trained weights.
- Safety labels are based on recipe text and heuristics, not independently verified food-allergy information.
- No authentication, permissions, distributed audit storage, rate limiting or production security hardening is included.
