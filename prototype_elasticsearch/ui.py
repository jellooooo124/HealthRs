"""Separate Streamlit ES recommendation tab. No LLM key required."""
from __future__ import annotations

import os
import requests
import streamlit as st

st.set_page_config(page_title="HealthRec · ES Prototype", page_icon="🥗", layout="wide")
API = os.environ.get("HEALTHREC_PROTOTYPE_API", "http://localhost:8000").rstrip("/")
CONDITIONS = ["diabetes", "hypertension", "hyperuricemia"]
ALLERGENS = ["shellfish", "milk", "peanut", "tree_nut", "fish", "egg", "wheat", "soy", "sesame"]


def request(method: str, path: str, **kwargs):
    response = requests.request(method, API + path, timeout=25, **kwargs)
    response.raise_for_status()
    return response.json()


st.title("🥗 HealthRec 2.0")
st.caption("Elasticsearch BM25 + existing BPR-MF (if available) + profile/feedback-aware personalized reranking")
user_id = st.sidebar.text_input("Demo user ID", "demo_alice")
try:
    profile = request("GET", f"/profiles/{user_id}")
except requests.RequestException as exc:
    st.error(f"Cannot reach prototype API at {API}: {exc}")
    st.stop()

with st.sidebar.form("profile"):
    st.subheader("Health profile")
    conditions = st.multiselect("Conditions", CONDITIONS, default=profile.get("conditions", []))
    allergies = st.multiselect("Allergies (hard exclusion)", ALLERGENS, default=profile.get("allergies", []))
    taste_text = st.text_input("Taste preferences", ", ".join(profile.get("taste_prefs", [])))
    if st.form_submit_button("Confirm profile", type="primary"):
        try:
            profile = request("PUT", f"/profiles/{user_id}", json={
                "conditions": conditions, "allergies": allergies,
                "taste_prefs": [s.strip() for s in taste_text.split(",") if s.strip()], "confirmed": True})
            st.session_state.pop("results", None)
            st.success("Profile saved")
        except requests.RequestException as exc:
            st.error(str(exc))

query = st.text_input("Recipe search", "quick chicken dinner")
col1, col2, col3 = st.columns(3)
with col1:
    minutes = st.number_input("Max minutes (0 = any)", 0, 360, 30)
with col2:
    calories = st.number_input("Max kcal (0 = any)", 0, 3000, 0)
with col3:
    k = st.slider("Recommendations", 1, 10, 5)

if st.button("🔎 Personalized recommend", type="primary"):
    try:
        response = request("POST", "/recommendations", json={
            "user_id": user_id, "query": query, "k": k,
            "constraints": {"max_minutes": minutes or None, "max_calories": calories or None}})
        st.session_state["results"] = response
    except requests.RequestException as exc:
        st.error(str(exc))

results = st.session_state.get("results")
if results and results.get("user_id") != user_id:
    results = None
if results:
    if results["status"] != "ok":
        st.warning(results.get("message", "Please confirm profile first."))
    else:
        st.caption(f"{results['n_candidates']} candidates · Original BPR used: {results['bpr_used']}")
        if not results["recipes"]:
            st.info("No recipe satisfies these constraints. Try a broader search; allergy/health checks remain enforced.")
        for recipe in results["recipes"]:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.subheader(recipe["name"])
                    st.write(" · ".join(recipe["reasons"]))
                    n = recipe["nutrition"]
                    st.caption(f"#{recipe['recipe_id']} · {recipe['category']} · {recipe.get('total_minutes')} min · "
                               f"{n.get('calories')} kcal · sodium {n.get('sodium_mg')} mg")
                    with st.expander("Explain ranking"):
                        st.json({"score": recipe["score"], "components": recipe["score_components"],
                                 "weights": recipe["score_weights"], "allergens_detected": recipe["allergens"]})
                with right:
                    for signal, label in (("like", "👍 Like"), ("dislike", "👎 Dislike")):
                        if st.button(label, key=f"{signal}_{user_id}_{recipe['recipe_id']}"):
                            try:
                                request("POST", "/feedback", json={"user_id": user_id,
                                    "recipe_id": recipe["recipe_id"], "signal": signal})
                                st.success("Feedback saved. Search again to update rankings.")
                            except requests.RequestException as exc:
                                st.error(str(exc))
        st.caption(results["note"])
