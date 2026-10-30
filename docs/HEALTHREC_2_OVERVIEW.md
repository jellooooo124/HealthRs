# HealthRec 2.0 — Project overview

**One-line summary (suggested GitHub About description):**

> Health-aware recipe recommender with Elasticsearch BM25, personalized BPR ranking, nutrition & allergy safeguards, feedback learning, LangGraph agents and RAG.

## Portfolio introduction

HealthRec is a health-aware, personalized recipe recommendation research platform for chronic-disease dietary contexts. Its original implementation combines PyTorch BPR matrix factorization, condition-aware ranking, allergen screening and nutrition labels with LangGraph multi-agent conversation, MCP tools, Chroma-based guideline retrieval and SQLite user profiles. HealthRec 2.0 extends the original system with Elasticsearch BM25 recipe retrieval, structured nutritional filtering, transparent hybrid scoring, a feedback-responsive personalized ranking path, and standalone FastAPI/Streamlit interfaces. The extension can optionally score Elasticsearch candidates with the original trained model. Its safety checks depend on heuristic labels and are not a substitute for clinical or allergy verification.

## What's implemented vs. planned

| Implemented | Future engineering work |
| --- | --- |
| ES indexer for bundled sample and prepared catalog | Route original MCP `recommend_recipes` through the ES retrieval channel |
| Query constraints and dual pre/post safety filtering | Authentication, consent and deployment-grade privacy/security |
| BM25 + taste/feedback/popularity hybrid ranking | Learn/calibrate hybrid weights on held-out interaction data |
| Optional baseline BPR scoring for warm users | Integrate vector / dense retrieval as a measured ablation |
| FastAPI + Streamlit demo, 8 offline tests | Live ES integration tests, full-catalog ranking and safety evaluation |

## Important evidence boundary

Original HealthRec's offline tables and agent evaluation belong to the baseline. This extension has only offline unit-test coverage reported here; it does not claim new NDCG/Recall gains, clinical safety or production readiness.

For full instructions see [the extension README](../prototype_elasticsearch/README.md) and the [repository README](../README.md).