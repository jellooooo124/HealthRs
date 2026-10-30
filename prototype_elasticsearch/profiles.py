"""Reuses HealthRec's profile/feedback schema, with a sample-only SQLite DB when needed."""
from __future__ import annotations

import os
from pathlib import Path
from agent.profile_store import ProfileStore

ROOT = Path(__file__).resolve().parents[1]


class OptionalSeedProfileStore(ProfileStore):
    def _seed(self, csv_path: str) -> None:
        # Existing HealthRec seeds its full-profile CSV; small-sample mode has no such CSV.
        if os.path.exists(csv_path):
            super()._seed(csv_path)


def make_store() -> ProfileStore:
    override = os.getenv("HEALTHREC_PROFILE_DB")
    if override:
        db = Path(override).expanduser().resolve()
    elif (ROOT / "data/processed/profiles.csv").exists():
        db = ROOT / "data/processed/app.db"
    else:
        db = ROOT / "prototype_elasticsearch/state/demo_profiles.db"
    return OptionalSeedProfileStore(db_path=str(db))
