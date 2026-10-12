"""
Multi-user profile store (SQLite, <processed_dir>/app.db — one database per dataset).

    user_profiles(user_id, source, user_idx, conditions, allergies, goals, taste_prefs, confirmed, updated_at)
    feedback(id, user_id, recipe_id, signal, value, reason, ts)

* Food.com users are seeded from data/processed/profiles.csv (synthetic health profiles, confirmed=1).
* Users created in the demo start empty with confirmed=0 (cold start); the profile agent asks for
  conditions/allergies before the first recommendation.
* Safety-relevant fields (conditions, allergies) are only ever read from here by the recommender tool,
  never taken from LLM-generated tool arguments.
"""
import json
import os
import sqlite3
import threading
import time

import pandas as pd

from src.config import get_project_root, load_config

_lock = threading.Lock()


class ProfileStore:
    def __init__(self, db_path: str | None = None):
        root = get_project_root()
        self.db_path = db_path or os.path.join(root, load_config("data")["paths"]["processed_dir"], "app.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        rules = load_config("health_rules")
        self.valid_conditions = set(rules["conditions"])
        self.valid_allergens = set(rules["allergens"])
        self.goal_text = rules["goals"]
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY, source TEXT, user_idx INTEGER,
                    conditions TEXT, allergies TEXT, goals TEXT, taste_prefs TEXT,
                    confirmed INTEGER DEFAULT 0, updated_at REAL);
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, recipe_id INTEGER,
                    signal TEXT, value REAL, reason TEXT, ts REAL);
                CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);
            """)
            n = c.execute("SELECT COUNT(*) FROM user_profiles WHERE source LIKE 'foodcom%'").fetchone()[0]
        if n == 0:
            self._seed(os.path.join(root, load_config("data")["paths"]["processed_dir"], "profiles.csv"))

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _seed(self, csv_path: str):
        df = pd.read_csv(csv_path, dtype={"user_idx": "Int64"})
        now = time.time()
        rows = [(str(r.user_id), r.source, None if pd.isna(r.user_idx) else int(r.user_idx), r.conditions, r.allergies,
                 r.goals, r.taste_prefs, 1, now) for r in df.itertuples(index=False)]
        with _lock, self._conn() as c:
            c.executemany("INSERT OR IGNORE INTO user_profiles VALUES (?,?,?,?,?,?,?,?,?)", rows)

    # ---------------- profiles
    def get(self, user_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM user_profiles WHERE user_id=?", (str(user_id),)).fetchone()
        if r is None:
            return None
        return {"user_id": r["user_id"], "source": r["source"], "user_idx": r["user_idx"],
                "conditions": json.loads(r["conditions"] or "[]"), "allergies": json.loads(r["allergies"] or "[]"),
                "goals": json.loads(r["goals"] or "[]"), "taste_prefs": json.loads(r["taste_prefs"] or "[]"),
                "confirmed": bool(r["confirmed"]), "updated_at": r["updated_at"]}

    def create(self, user_id: str) -> dict:
        with _lock, self._conn() as c:
            c.execute("INSERT OR IGNORE INTO user_profiles VALUES (?,?,?,?,?,?,?,?,?)",
                      (str(user_id), "demo_new", None, "[]", "[]", "[]", "[]", 0, time.time()))
        return self.get(user_id)

    def update(self, user_id: str, conditions: list | None = None, allergies: list | None = None,
               taste_prefs: list | None = None, confirmed: bool | None = None) -> dict:
        """Replace the given fields (unknown condition/allergen names are dropped). Goals follow conditions."""
        p = self.get(user_id) or self.create(user_id)
        if conditions is not None:
            p["conditions"] = sorted({c for c in conditions if c in self.valid_conditions})
            p["goals"] = [self.goal_text[c] for c in p["conditions"]]
        if allergies is not None:
            p["allergies"] = sorted({a for a in allergies if a in self.valid_allergens})
        if taste_prefs is not None:
            p["taste_prefs"] = list(dict.fromkeys(t.strip() for t in taste_prefs if t and t.strip()))[:10]
        if confirmed is not None:
            p["confirmed"] = confirmed
        with _lock, self._conn() as c:
            c.execute("UPDATE user_profiles SET conditions=?, allergies=?, goals=?, taste_prefs=?, confirmed=?, updated_at=? "
                      "WHERE user_id=?", (json.dumps(p["conditions"]), json.dumps(p["allergies"]), json.dumps(p["goals"]),
                                          json.dumps(p["taste_prefs"]), int(p["confirmed"]), time.time(), str(user_id)))
        return self.get(user_id)

    def list_users(self, source_prefix: str = "", limit: int = 200) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT user_id FROM user_profiles WHERE source LIKE ? ORDER BY updated_at DESC LIMIT ?",
                             (source_prefix + "%", limit)).fetchall()
        return [self.get(r["user_id"]) for r in rows]

    # ---------------- feedback
    def add_feedback(self, user_id: str, recipe_id: int, signal: str, reason: str = "", value: float = 1.0):
        with _lock, self._conn() as c:
            c.execute("INSERT INTO feedback (user_id, recipe_id, signal, value, reason, ts) VALUES (?,?,?,?,?,?)",
                      (str(user_id), int(recipe_id), signal, value, reason, time.time()))

    def feedback(self, user_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT recipe_id, signal, value, reason, ts FROM feedback WHERE user_id=? ORDER BY ts",
                             (str(user_id),)).fetchall()
        return [dict(r) for r in rows]

    def latest_feedback(self, user_id: str) -> dict[int, dict]:
        """Current feedback per recipe = its latest signal, most recent last; a "clear" signal removes it."""
        out = {}
        for f in self.feedback(user_id):   # ordered by time
            out.pop(f["recipe_id"], None)
            if f["signal"] != "clear":
                out[f["recipe_id"]] = f
        return out

    def disliked(self, user_id: str) -> set[int]:
        return {rid for rid, f in self.latest_feedback(user_id).items() if f["signal"] == "dislike"}

    def clone_user(self, src_id: str | None, new_id: str, confirmed: bool | None = None,
                   blank_health: bool = False) -> dict:
        """
        (Re)create `new_id` as a copy of `src_id` (same source/user_idx/profile) with no feedback — used by the
        agent evaluation so every (config, scenario) dialog has its own isolated user. blank_health=True makes a
        cold-start user whose conditions/allergies the agent must ask for.
        """
        src = self.get(src_id) if src_id else None
        with _lock, self._conn() as c:
            c.execute("DELETE FROM feedback WHERE user_id=?", (new_id,))
            c.execute("DELETE FROM user_profiles WHERE user_id=?", (new_id,))
            c.execute("INSERT INTO user_profiles VALUES (?,?,?,?,?,?,?,?,?)", (
                new_id, (src or {}).get("source", "demo_new"), (src or {}).get("user_idx"),
                "[]" if blank_health or not src else json.dumps(src["conditions"]),
                "[]" if blank_health or not src else json.dumps(src["allergies"]),
                "[]" if blank_health or not src else json.dumps(src["goals"]),
                json.dumps((src or {}).get("taste_prefs", [])),
                int(confirmed if confirmed is not None else (bool(src) and not blank_health)), time.time()))
        return self.get(new_id)

    def reset_demo_user(self, user_id: str):
        with _lock, self._conn() as c:
            c.execute("DELETE FROM feedback WHERE user_id=?", (str(user_id),))
            c.execute("DELETE FROM user_profiles WHERE user_id=? AND source='demo_new'", (str(user_id),))
