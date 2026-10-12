"""
HealthRec demo — Streamlit UI for the health-aware recipe recommender agent.

    streamlit run web/app_ui.py

Sidebar: switch between demo users / create a new (cold-start) user, view and edit the health profile,
see feedback, upload guideline documents to the knowledge base.
Main: chat with the 3-agent pipeline; recommended recipes are shown with per-serving nutrition,
condition compliance and like/dislike buttons; "How this answer was made" shows the agent trace.
"""
import html
import json
import os
import re
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.profile_store import ProfileStore  # noqa: E402
from agent.runtime import SyncHealthRecAgent  # noqa: E402
from src.config import get_agent_config, get_project_root, load_config  # noqa: E402

st.set_page_config(page_title="HealthRec", page_icon="🥗", layout="wide")

CONDITIONS = {"diabetes": "Type 2 diabetes", "hypertension": "Hypertension", "hyperuricemia": "High uric acid / gout"}
ALLERGENS = {"shellfish": "Shellfish", "milk": "Milk", "peanut": "Peanut", "tree_nut": "Tree nuts", "fish": "Fish",
             "egg": "Egg", "wheat": "Wheat", "soy": "Soy", "sesame": "Sesame"}
NUTRI_COLS = [("kcal", "kcal"), ("carbs_g", "Carbs (g)"), ("sugar_g", "Sugar (g)"), ("sodium_mg", "Sodium (mg)"),
              ("sat_fat_g", "Sat. fat (g)"), ("protein_g", "Protein (g)"), ("fiber_g", "Fiber (g)")]
SUGGESTIONS = ["Quick dinner ideas under 30 minutes", "A breakfast that's gentle on my blood sugar",
               "Something sweet that is still OK for me", "How much sodium per day is OK for me?"]
SHORT = ["⏱ Quick dinner", "🍳 Blood-sugar breakfast", "🍪 Something sweet", "🧂 Sodium per day?"]
UI_KEYS = ("recipes", "trace", "usage", "route", "revisions", "latency_s", "conditions")


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Starting the agents and the recommender service (first run ~10 s)…")
def get_agent() -> SyncHealthRecAgent:
    return SyncHealthRecAgent()


@st.cache_resource
def get_store() -> ProfileStore:
    return ProfileStore()


@st.cache_resource
def recipe_meta() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(get_project_root(), "data", "processed", "recipes.parquet"),
                         columns=["RecipeId", "Name", "ImageUrl"])
    df["Name"] = df["Name"].map(html.unescape)
    return df.set_index("RecipeId")


agent, store, cfg = get_agent(), get_store(), get_agent_config()
ss = st.session_state
ss.setdefault("user_id", cfg["demo"]["users"][0]["id"])
ss.setdefault("turns", {})


def ui_path(uid: str) -> str:
    """Sidecar file with the cards / nutrition / trace of each assistant turn (the agent memory stores text only)."""
    return os.path.join(agent.agent.memory.memory_dir, agent.agent._session(uid) + ".ui.json")


def load_ui(uid: str) -> list:
    try:
        with open(ui_path(uid)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save_ui(uid: str, entry: dict):
    keep = get_agent_config()["state"]["max_history_messages"]
    items = (load_ui(uid) + [entry])[-keep:]
    with open(ui_path(uid), "w") as f:
        json.dump(items, f, ensure_ascii=False)


def turns_for(uid: str) -> list:
    if uid not in ss.turns:   # restore earlier sessions: text from the agent memory + cards/trace from the sidecar
        hist = agent.agent.memory.load_history(agent.agent._session(uid))
        turns = [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in hist]
        ui = load_ui(uid)
        assistant = [t for t in turns if t["role"] == "assistant"]
        for t, meta in zip(reversed(assistant), reversed(ui)):   # both are trimmed from the front: align from the end
            if meta.get("content") == t["content"]:
                t.update({k: meta[k] for k in UI_KEYS if k in meta})
        ss.turns[uid] = turns
    return ss.turns[uid]


def clear_conversation(uid: str):
    agent.agent.reset_memory(uid)
    try:
        os.remove(ui_path(uid))
    except OSError:
        pass
    ss.turns[uid] = []


def chips(items: list, mapping: dict, empty: str) -> str:
    if not items:
        return f"<span style='color:var(--text-color);opacity:.6'>{empty}</span>"
    return " ".join(f"<span style='background:rgba(42,120,214,.12);border-radius:12px;padding:2px 10px;"
                    f"margin-right:4px;font-size:.85em'>{mapping.get(x, x)}</span>" for x in items)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🥗 Who is cooking?")
    demo = cfg["demo"]["users"]
    options = [f"{u['label']}  (#{u['id']})" for u in demo]
    current = next((i for i, u in enumerate(demo) if u["id"] == ss.user_id), None)
    if current is None:   # a created or looked-up user stays selectable
        options.append(f"👤 {ss.user_id}")
        current = len(options) - 1
    options += ["➕ New user (cold start)", "🔎 Another Food.com user"]
    choice = st.selectbox("User", options, index=current)
    if choice.startswith("👤"):
        pass
    elif choice.startswith("➕"):
        name = st.text_input("Your name", placeholder="e.g. Alice")
        if st.button("Create profile", use_container_width=True) and name.strip():
            uid = "new_" + re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
            store.get(uid) or store.create(uid)
            ss.user_id = uid
            st.rerun()
    elif choice.startswith("🔎"):
        uid = st.text_input("Food.com user id", placeholder="e.g. 1533")
        if st.button("Open", use_container_width=True) and uid.strip():
            if store.get(uid.strip()):
                ss.user_id = uid.strip()
                st.rerun()
            st.error("Unknown user id")
    else:
        picked = demo[options.index(choice)]["id"]
        if picked != ss.user_id:
            ss.user_id = picked
            st.rerun()

    uid = ss.user_id
    profile = store.get(uid) or store.create(uid)
    st.subheader("Health profile")
    st.caption(f"User `{uid}` · " + ("rating history on Food.com" if profile["source"] == "foodcom_warm"
                                     else "no rating history — cold start"))
    st.markdown("**Conditions**<br>" + chips(profile["conditions"], CONDITIONS, "none recorded"), unsafe_allow_html=True)
    st.markdown("**Food allergies**<br>" + chips(profile["allergies"], ALLERGENS, "none recorded"), unsafe_allow_html=True)
    if profile["taste_prefs"]:
        st.markdown("**Taste**<br>" + chips(profile["taste_prefs"], {}, ""), unsafe_allow_html=True)
    if not profile["confirmed"]:
        st.info("Profile not confirmed yet — the assistant will ask about conditions and allergies first.")

    with st.expander("✏️ Edit profile"):
        with st.form("edit_profile"):
            conds = st.multiselect("Conditions", list(CONDITIONS), profile["conditions"], format_func=CONDITIONS.get)
            algs = st.multiselect("Food allergies", list(ALLERGENS), profile["allergies"], format_func=ALLERGENS.get)
            prefs = st.text_input("Taste preferences (comma-separated)", ", ".join(profile["taste_prefs"]))
            if st.form_submit_button("Save", use_container_width=True):
                store.update(uid, conds, algs, [p for p in prefs.split(",")], confirmed=True)
                st.rerun()

    fb = list(store.latest_feedback(uid).values())[-6:]
    if fb:
        st.markdown("**Recent feedback**")
        names = recipe_meta()["Name"]
        for f in reversed(fb):
            icon = {"like": "👍", "dislike": "👎", "cooked": "🍳"}.get(f["signal"], "•")
            st.caption(f"{icon} {names.get(f['recipe_id'], '#' + str(f['recipe_id']))}"
                       + (f" — {f['reason']}" if f["reason"] else ""))

    with st.popover("🗑 Clear conversation", use_container_width=True):
        st.markdown("Permanently delete this user's chat history (text, recipe cards and traces)? "
                    "The health profile and feedback are kept.")
        if st.button("Yes, delete history", type="primary", use_container_width=True):
            clear_conversation(uid)
            st.rerun()

    with st.expander("📚 Diet-guideline knowledge base"):
        gdir = os.path.join(get_project_root(), "data", "guidelines")
        st.caption("Indexed: " + ", ".join(sorted(f for f in os.listdir(gdir) if f.endswith((".md", ".txt", ".pdf")))))
        up = st.file_uploader("Add a guideline document", type=["md", "txt", "pdf"])
        if up is not None and st.button("Index document", use_container_width=True):
            os.makedirs(os.path.join(gdir, "uploads"), exist_ok=True)
            path = os.path.join(gdir, "uploads", os.path.basename(up.name))
            with open(path, "wb") as f:
                f.write(up.getbuffer())
            st.success(agent.call_tool("ingest_guideline_document", path=path))

    st.divider()
    pipe = cfg["pipeline"]
    fb_note = (" (fallback " + " → ".join(f"`{f}`" for f in agent.agent.fallback_names) + ")"
               if agent.agent.fallback_names else "")
    st.caption(f"LLM: `{agent.agent.model_name}`{fb_note} · recommender: BPR-MF + nutrition/condition features "
               f"(popularity-calibrated) · reviewer {'on' if pipe['safety_reviewer'] else 'off'} · "
               f"clarification {'on' if pipe['clarification'] else 'off'}")


# ------------------------------------------------------------------ main area
def recipe_block(turn_idx: int, turn: dict):
    recs, conds = turn.get("recipes") or [], turn.get("conditions", [])
    if not recs:
        return
    per_row = 3
    current = {rid: f["signal"] for rid, f in store.latest_feedback(ss.user_id).items()}
    for j, r in enumerate(recs[:6]):
        if j % per_row == 0:
            cols = st.columns(per_row)
        with cols[j % per_row]:
            with st.container(border=True):
                img = recipe_meta()["ImageUrl"].get(r["recipe_id"])
                if isinstance(img, str) and img.startswith("http"):
                    st.image(img, use_container_width=True)
                st.markdown(f"**{r['name']}**")
                mins = f"{r['total_minutes']} min" if r.get("total_minutes") else "time n/a"
                st.caption(f"#{r['recipe_id']} · {r['category']} · {mins}")
                badges = [("✅ " if ok else "⚠️ ") + CONDITIONS[c].split(" /")[0] for c, ok in r["complies_with"].items()]
                if badges:
                    st.caption(" · ".join(badges))
                b1, b2 = st.columns(2)
                liked, disliked = current.get(r["recipe_id"]) == "like", current.get(r["recipe_id"]) == "dislike"
                # the selected button is highlighted (primary = red); clicking it again removes the feedback
                if b1.button("", icon=":material/thumb_up:", type="primary" if liked else "secondary",
                             help="Remove like" if liked else "I like this",
                             key=f"like_{turn_idx}_{r['recipe_id']}", use_container_width=True):
                    store.add_feedback(ss.user_id, r["recipe_id"], "clear" if liked else "like", "thumbs up in demo")
                    st.toast(f"Removed like: {r['name']}" if liked else f"Saved: you like {r['name']}")
                    st.rerun()
                if b2.button("", icon=":material/thumb_down:", type="primary" if disliked else "secondary",
                             help="Recommend this again" if disliked else "Don't recommend this again",
                             key=f"dislike_{turn_idx}_{r['recipe_id']}", use_container_width=True):
                    store.add_feedback(ss.user_id, r["recipe_id"], "clear" if disliked else "dislike", "thumbs down in demo")
                    st.toast(f"{r['name']} can be recommended again" if disliked
                             else f"Saved: {r['name']} won't be recommended again")
                    st.rerun()
    rows = []
    for r in recs:
        n = r["nutrition_per_serving"]
        row = {"Recipe": r["name"], **{label: n.get(key) for key, label in NUTRI_COLS}}
        row["Fits your conditions"] = "✅ yes" if not r.get("violations") else "⚠️ " + "; ".join(
            f"{c}: {v}" for c, v in r["violations"].items())
        rows.append(row)
    with st.expander(f"📊 Nutrition per serving ({len(recs)} recipes)"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        rules = load_config("health_rules")["conditions"]
        st.caption("Limits per serving — " + "; ".join(
            f"{CONDITIONS[c]}: " + ", ".join(f"{k.replace('Content', '')} ≤ {v}" for k, v in rules[c]["nutrient_max"].items())
            for c in conds) if conds else "No conditions recorded: recipes are ranked by taste only (allergens still excluded).")


def trace_block(turn: dict):
    tr, u = turn.get("trace") or [], turn.get("usage") or {}
    with st.expander("🔍 How this answer was made"):
        st.caption(f"Route: **{turn.get('route')}** · revisions: {turn.get('revisions', 0)} · "
                   f"{turn.get('latency_s', 0)} s · {u.get('llm_calls', 0)} LLM calls · {u.get('total_tokens', 0):,} tokens")
        for step in tr:
            node = step.get("node")
            if node in ("load_profile", "finalize"):
                continue
            line = f"**{node}**" + (f" ({step['ms']} ms)" if "ms" in step else "")
            if node == "profile_agent":
                d = step.get("decision", {})
                line += f" — intent `{d.get('intent')}`, constraints " + str(
                    {k: v for k, v in d.get("constraints", {}).items() if v not in (None, [], 5)})
            if node == "safety_reviewer":
                rv = step["review"]
                line += " — " + ("✅ passed" if rv["passed"] else "❌ sent back: " + " | ".join(rv["issues"][:3]))
            st.markdown(line)
            for t in step.get("tools", []):
                st.caption(f"↳ tool `{t['tool']}` {t['args']} · {t['ms']} ms")


st.title("🥗 HealthRec")
st.caption("A health-aware recipe assistant for people managing diabetes, high blood pressure or gout: "
           "three agents (profile → recommender → safety reviewer) on top of a recommender trained on Food.com ratings. "
           "Suggestions follow public diet guidelines but are **not medical advice**.")

turns = turns_for(ss.user_id)
for i, t in enumerate(turns):
    with st.chat_message(t["role"], avatar="🧑" if t["role"] == "user" else "🥗"):
        st.markdown(t["content"])
        if t["role"] == "assistant" and "trace" in t:
            recipe_block(i, t)
            trace_block(t)

prompt = st.chat_input("Ask for recipes, give feedback, or ask a diet question…")
if not turns:
    st.markdown("**Try:**")
    for row in range(0, len(SUGGESTIONS), 2):
        for c, s in zip(st.columns(2), SUGGESTIONS[row:row + 2]):
            if c.button(s, use_container_width=True):
                prompt = s
else:   # compact row above the input box once a conversation is under way
    for c, label, s in zip(st.columns(len(SUGGESTIONS)), SHORT, SUGGESTIONS):
        if c.button(label, help=s, use_container_width=True, key=f"sugg_{label}"):
            prompt = s

if prompt:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🥗"):
        with st.spinner("Profile agent → recommender agent → safety reviewer…"):
            out = agent.chat(ss.user_id, prompt)
    cited = [int(x) for x in re.findall(r"#(\d{2,7})\b", out["reply"])]
    by_id = {r["recipe_id"]: r for r in out.get("candidates", [])}
    shown = [by_id[i] for i in dict.fromkeys(cited) if i in by_id]
    entry = {"role": "assistant", "content": out["reply"], "recipes": shown, "trace": out["trace"],
             "usage": out["usage"], "route": out["route"], "revisions": out["revisions"],
             "latency_s": out["latency_s"], "conditions": (out.get("profile") or profile)["conditions"]}
    turns.append({"role": "user", "content": prompt})   # appended only once the turn completed
    turns.append(entry)
    save_ui(ss.user_id, entry)
    st.rerun()
