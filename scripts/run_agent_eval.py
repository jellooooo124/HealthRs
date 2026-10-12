"""
Simulated-user evaluation of the agent (step 7).

    # pilot with the development model (debug the harness, measure tokens per turn)
    python scripts/run_agent_eval.py --run pilot --role agent --configs full llm_only --limit 2
    # formal run: agent under test pinned to qwen3.8-max-0902 (role "eval"), temperature 0, budget cap
    python scripts/run_agent_eval.py --run final --role eval --configs full llm_only no_reviewer no_clarify
    python scripts/run_agent_eval.py --run final --stage judge
    python scripts/run_agent_eval.py --run final --stage report

Configurations: full system | llm_only (pure-LLM baseline: no recommend_recipes tool) | no_reviewer |
no_clarify (cold-start scenarios only: on warm users clarification never triggers, so results would equal full).
Every LLM call is cached (outputs/agent_eval/llm_cache.sqlite): reruns and crashed runs cost nothing twice.
Token usage per model is accumulated in outputs/agent_eval/<run>/ledger.json; dialogs stop once the agent
model reaches --budget tokens.
"""
import argparse
import asyncio
import glob
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("HEALTHREC_MCP_PORT", "8766")      # separate MCP server in evaluation mode
os.environ.setdefault("HEALTHREC_EVAL_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from langchain_community.cache import SQLiteCache  # noqa: E402
from langchain_core.globals import set_llm_cache  # noqa: E402

from agent.models.cloud_model import UsageTracker  # noqa: E402
from agent.profile_store import ProfileStore  # noqa: E402
from agent.runtime import HealthRecAgent  # noqa: E402
from src.config import get_agent_config, get_project_root, get_outputs_dir  # noqa: E402
from src.eval.agent_eval.facts import RecipeFacts  # noqa: E402
from src.eval.agent_eval.judge import Judge  # noqa: E402
from src.eval.agent_eval.scenarios import build_scenarios  # noqa: E402
from src.eval.agent_eval.simulator import SimulatedUser  # noqa: E402

ROOT = get_project_root()
CONFIGS = {"full": {}, "llm_only": {"recommender_tool": False}, "no_reviewer": {"safety_reviewer": False},
           "no_clarify": {"clarification": False}}
MAX_TURNS = 3


def run_dir(run: str) -> str:
    d = get_outputs_dir("agent_eval", run)
    os.makedirs(d, exist_ok=True)
    return d


class Ledger:
    def __init__(self, path: str):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def add(self, tracker: UsageTracker):
        for model, v in tracker.by_model.items():
            m = self.data.setdefault(model, {"calls": 0, "tokens": 0})
            m["calls"] += v["calls"]
            m["tokens"] += v["tokens"]
        json.dump(self.data, open(self.path, "w"), indent=2)

    def tokens(self, model: str) -> int:
        return self.data.get(model, {}).get("tokens", 0)


def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


# ---------------------------------------------------------------- stage: dialogs
async def run_dialogs(args):
    d = run_dir(args.run)
    get_agent_config()["guardrails"]["llm_check"] = False          # planned saving: no moderation call in evaluation
    scenarios = build_scenarios()[: args.limit or None]
    if args.ids:
        scenarios = [s for s in build_scenarios() if s["id"] in args.ids]
    json.dump(scenarios, open(os.path.join(d, "scenarios.json"), "w"), indent=2)
    out_path, ledger = os.path.join(d, "dialogs.jsonl"), Ledger(os.path.join(d, "ledger.json"))
    done = {(r["config"], r["scenario"]) for r in load_jsonl(out_path)}
    store = ProfileStore()
    agents = {c: await HealthRecAgent.create(role=args.role, pipeline_overrides=CONFIGS[c],
                                             memory_dir=os.path.join(d, "memory", c)) for c in args.configs}
    model = next(iter(agents.values())).model_name
    jobs = [(c, s) for s in scenarios for c in args.configs
            if (c, s["id"]) not in done and not (c == "no_clarify" and s["kind"] != "cold")]
    print(f"{len(jobs)} dialogs to run with agent model {model}; ledger so far: {ledger.data}", flush=True)
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()

    async def one(cfg_name: str, sc: dict):
        async with sem:
            if ledger.tokens(model) >= args.budget:
                print(f"budget reached for {model}; skipping {cfg_name}/{sc['id']}", flush=True)
                return
            uid = f"ev_{args.run}_{cfg_name}_{sc['id']}"
            store.clone_user(sc["source_user"], uid, blank_health=sc["kind"] == "cold")
            agent = agents[cfg_name]
            agent.reset_memory(uid)
            sim, sim_usage = SimulatedUser(sc, MAX_TURNS), UsageTracker()
            msg, transcript, turns, t0 = sim.opener(), [], [], time.time()
            for t in range(1, MAX_TURNS + 1):
                out = await agent.chat(uid, msg)
                transcript += [{"role": "user", "content": msg}, {"role": "assistant", "content": out["reply"]}]
                verdict = await sim.reply(transcript, t + 1, callbacks=[sim_usage])
                turns.append({"turn": t, "user": msg, "reply": out["reply"], "route": out["route"],
                              "revisions": out["revisions"], "latency_s": out["latency_s"], "usage": out["usage"],
                              "candidates": [r["recipe_id"] for r in out.get("candidates", [])],
                              "request": out.get("request"), "sim": verdict,
                              "trace": [{k: v for k, v in tr.items() if k in ("node", "ms", "review", "decision")}
                                        for tr in out["trace"]]})
                async with lock:
                    tr = UsageTracker()
                    tr.by_model = out["usage"]["by_model"]
                    ledger.add(tr)
                if verdict["satisfied"] or verdict["end"]:
                    break
                msg = verdict["message"]
            async with lock:
                ledger.add(sim_usage)
            rec = {"run": args.run, "config": cfg_name, "scenario": sc["id"], "kind": sc["kind"], "user_id": uid,
                   "agent_model": model, "transcript": transcript, "turns": turns,
                   "success": bool(turns[-1]["sim"]["satisfied"]), "seconds": round(time.time() - t0, 1)}
            async with lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{cfg_name}/{sc['id']}] turns={len(turns)} success={rec['success']} "
                      f"agent_tokens={sum(t['usage']['total_tokens'] for t in turns)} ledger[{model}]={ledger.tokens(model)}",
                      flush=True)

    await asyncio.gather(*(one(c, s) for c, s in jobs))


# ---------------------------------------------------------------- stage: judge
async def run_judge(args):
    d = run_dir(args.run)
    scen = {s["id"]: s for s in json.load(open(os.path.join(d, "scenarios.json")))}
    dialogs = load_jsonl(os.path.join(d, "dialogs.jsonl"))
    out_path = os.path.join(d, "judge.jsonl")
    ok = [r for r in load_jsonl(out_path) if "error" not in r]          # failed judgments are redone
    with open(out_path, "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in ok)
    done = {(r["config"], r["scenario"]) for r in ok}
    facts, judge, sem = RecipeFacts(), Judge(), asyncio.Semaphore(args.concurrency)
    ledger, lock = Ledger(os.path.join(d, "ledger.json")), asyncio.Lock()

    async def one(rec):
        async with sem:
            sc = scen[rec["scenario"]]
            cited = list(dict.fromkeys(i for t in rec["turns"] for i in facts.cited(t["reply"])))
            f = [facts.check(i, sc["conditions"], sc["allergies"], sc["dislike"], sc["time_limit"]) for i in cited]
            usage = UsageTracker()
            res = await judge.rate(sc, rec["transcript"], f, callbacks=[usage])
            async with lock:
                ledger.add(usage)
            with open(out_path, "a") as fh:
                fh.write(json.dumps({"config": rec["config"], "scenario": rec["scenario"], **res}, ensure_ascii=False) + "\n")
            print(f"judged {rec['config']}/{rec['scenario']}: {({k: res.get(k) for k in ('relevance', 'faithfulness', 'safety', 'overall')})}", flush=True)

    await asyncio.gather(*(one(r) for r in dialogs if (r["config"], r["scenario"]) not in done))


# ---------------------------------------------------------------- stage: report
def wilson(k: int, n: int) -> str:
    if n == 0:
        return "–"
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.0%} [{max(0, c - h):.0%}, {min(1, c + h):.0%}]"


def report(args):
    d = run_dir(args.run)
    scen = {s["id"]: s for s in json.load(open(os.path.join(d, "scenarios.json")))}
    dialogs = load_jsonl(os.path.join(d, "dialogs.jsonl"))
    judged = {(r["config"], r["scenario"]): r for r in load_jsonl(os.path.join(d, "judge.jsonl"))}
    facts = RecipeFacts()
    rows = []
    for rec in dialogs:
        sc, turns = scen[rec["scenario"]], rec["turns"]
        cited_all, final = [], facts.cited(turns[-1]["reply"])
        for t in turns:
            cited_all += facts.cited(t["reply"])
        checks = [facts.check(i, sc["conditions"], sc["allergies"], sc["dislike"], sc["time_limit"]) for i in dict.fromkeys(cited_all)]
        known = [c for c in checks if c["known"]]
        fin = [facts.check(i, sc["conditions"], sc["allergies"], sc["dislike"], sc["time_limit"]) for i in final]
        fin = [c for c in fin if c["known"]]
        rec_turns = [t for t in turns if t["route"] != "clarify"]
        j = judged.get((rec["config"], rec["scenario"]), {})
        rows.append({
            "config": rec["config"], "scenario": rec["scenario"], "kind": rec["kind"], "success": rec["success"],
            "turns": len(turns), "clarified_first": turns[0]["route"] == "clarify",
            "n_cited": len(checks), "unknown_ids": sum(not c["known"] for c in checks),
            "violating": sum(bool(c["violations"]) for c in known), "allergen": sum(bool(c["allergens"]) for c in known),
            "final_dislike": sum(c["contains_dislike"] for c in fin), "final_over_time": sum(c["over_time"] for c in fin),
            "n_final": len(fin), "revised_turns": sum(t["revisions"] > 0 for t in turns),
            "misattributed": sum(len(facts.misattributed(t["reply"])) for t in turns),
            "rejected_drafts": sum(1 for t in turns for x in t["trace"] if x["node"] == "safety_reviewer" and not x["review"]["passed"]),
            "fallback_turns": sum(t["route"] == "fallback" for t in turns),
            "tokens_per_turn": np.mean([t["usage"]["total_tokens"] for t in turns]),
            "calls_per_turn": np.mean([t["usage"]["llm_calls"] for t in turns]),
            "latency_per_turn": np.mean([t["latency_s"] for t in turns]),
            "rec_turn_tokens": np.mean([t["usage"]["total_tokens"] for t in rec_turns]) if rec_turns else np.nan,
            **{k: j.get(k) for k in ("relevance", "faithfulness", "safety", "overall")},
            "judge_unsafe_items": len(j.get("unsafe_items") or []) if j else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "per_dialog.csv"), index=False)

    def summarize(g: pd.DataFrame) -> dict:
        cited = g["n_cited"].sum()
        return {
            "dialogs": len(g), "task success [95% CI]": wilson(int(g["success"].sum()), len(g)),
            "avg turns": round(g["turns"].mean(), 2),
            "recipes cited (with id)": int(cited),
            "violating share": f"{g['violating'].sum() / cited:.1%}" if cited else "n/a (no ids)",
            "allergen share": f"{g['allergen'].sum() / cited:.1%}" if cited else "n/a (no ids)",
            "unknown ids": int(g["unknown_ids"].sum()),
            "misattributed citations (name ≠ id) in replies": int(g["misattributed"].sum()),
            "drafts rejected by reviewer": int(g["rejected_drafts"].sum()),
            "judge: dialogs with unsafe items": f"{(g['judge_unsafe_items'] > 0).mean():.0%}" if g["judge_unsafe_items"].notna().any() else "–",
            **{f"judge {k}": f"{g[k].mean():.2f} ± {g[k].std(ddof=0):.2f}" if g[k].notna().any() else "–"
               for k in ("relevance", "faithfulness", "safety", "overall")},
            "turns revised": f"{g['revised_turns'].sum() / g['turns'].sum():.0%}",
            "tokens/turn": int(g["tokens_per_turn"].mean()), "LLM calls/turn": round(g["calls_per_turn"].mean(), 1),
            "latency/turn (s)": round(g["latency_per_turn"].mean(), 1),
        }

    order = [c for c in CONFIGS if c in set(df["config"])]
    main = pd.DataFrame([{"config": c, **summarize(df[df["config"] == c])} for c in order])
    cold = pd.DataFrame([{"config": c, **summarize(df[(df["config"] == c) & (df["kind"] == "cold")]),
                          "asked health first": f"{df[(df['config'] == c) & (df['kind'] == 'cold')]['clarified_first'].mean():.0%}"}
                         for c in order if len(df[(df["config"] == c) & (df["kind"] == "cold")])])
    # paired comparison with the full system on the same scenarios (Wilcoxon signed-rank on judge scores)
    from scipy.stats import wilcoxon
    tests = []
    base = df[df["config"] == "full"].set_index("scenario")
    for c in order:
        if c == "full":
            continue
        other = df[df["config"] == c].set_index("scenario")
        common = base.index.intersection(other.index)
        row = {"comparison": f"full vs {c}", "paired dialogs": len(common)}
        for k in ("relevance", "faithfulness", "safety", "overall"):
            a, b = base.loc[common, k].astype(float), other.loc[common, k].astype(float)
            ok = a.notna() & b.notna()
            diff = (a - b)[ok]
            p = wilcoxon(a[ok], b[ok]).pvalue if (diff != 0).any() else 1.0
            row[k] = f"{diff.mean():+.2f} (p={p:.3f})"
        tests.append(row)
    tests = pd.DataFrame(tests)
    tab = get_outputs_dir("tables")
    tests.to_csv(os.path.join(tab, f"agent_eval_{args.run}_tests.csv"), index=False)
    main.to_csv(os.path.join(tab, f"agent_eval_{args.run}.csv"), index=False)
    cold.to_csv(os.path.join(tab, f"agent_eval_{args.run}_cold.csv"), index=False)
    ledger = json.load(open(os.path.join(d, "ledger.json"))) if os.path.exists(os.path.join(d, "ledger.json")) else {}

    def md(t: pd.DataFrame) -> str:
        t = t.set_index("config").T.reset_index().rename(columns={"index": "metric"})
        lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
        return "\n".join(lines + ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)])

    text = [f"# Agent evaluation — run `{args.run}`\n",
            f"Simulated users: {len(scen)} scenarios (warm Food.com users with conditions + cold-start users), "
            f"≤{MAX_TURNS} assistant turns. Agent model: {dialogs[0]['agent_model'] if dialogs else '?'} (temperature 0); "
            "simulated user: deepseek-v4.1-flash; judge: kimi-k3. `no_clarify` runs on cold-start scenarios only.\n",
            "## All scenarios\n", md(main), "", "## Cold-start scenarios\n", md(cold) if len(cold) else "(none)", "",
            "## Paired tests vs full system (mean judge-score difference full − other; Wilcoxon signed-rank)\n",
            "| " + " | ".join(tests.columns) + " |\n|" + "---|" * len(tests.columns) + "\n" + "\n".join(
                "| " + " | ".join(str(v) for v in r) + " |" for r in tests.itertuples(index=False)), "",
            "## Judge reliability (human check on a stratified sample)\n",
            "\n\n".join("Team members rated a stratified sample of dialogs blind (without seeing the judge's scores); "
                         "agreement with the LLM judge:\n\n" + md_plain_tab(pd.read_csv(p))
                         for p in sorted(glob.glob(os.path.join(tab, f"agent_eval_{args.run}_agreement_human_check.csv")))) or "(no human ratings yet)", "",
            "## Token ledger (all calls of this run, incl. cached replays)\n", "```", json.dumps(ledger, indent=2), "```"]
    with open(os.path.join(tab, f"agent_eval_{args.run}.md"), "w") as f:
        f.write("\n".join(text) + "\n")
    print("\n".join(text))
    human_check(args.run, dialogs, judged, scen)


def md_plain_tab(t: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)])


def human_check(run: str, dialogs: list, judged: dict, scen: dict, n: int = 12):
    """Stratified sample of dialogs for manual verification of the judge's scores."""
    random.seed(2026)
    by_cfg = {}
    for r in dialogs:
        by_cfg.setdefault(r["config"], []).append(r)
    sample = []
    while len(sample) < n and any(by_cfg.values()):
        for c in list(by_cfg):
            if by_cfg[c] and len(sample) < n:
                sample.append(by_cfg[c].pop(random.randrange(len(by_cfg[c]))))
    d = run_dir(run)
    lines = ["# Human check of judge scores\n",
             "For each dialog, read the transcript, write your own 1–5 scores (relevance, faithfulness, safety, overall) "
             "in human_check.csv, then run `python scripts/run_agent_eval.py --run " + run + " --stage agreement`.\n"]
    rows = []
    for r in sample:
        sc, j = scen[r["scenario"]], judged.get((r["config"], r["scenario"]), {})
        lines += [f"## {r['config']} / {r['scenario']}",
                  f"Persona: conditions {sc['conditions']}, allergies {sc['allergies']}, wants {sc['meal']}, "
                  f"≤{sc['time_limit']} min, dislikes {sc['dislike']}", ""]
        lines += [f"**{m['role'].upper()}:** {m['content']}\n" for m in r["transcript"]]
        lines += [f"Judge: relevance {j.get('relevance')}, faithfulness {j.get('faithfulness')}, safety {j.get('safety')}, "
                  f"overall {j.get('overall')} — {j.get('rationale', '')}", "", "---", ""]
        rows.append({"config": r["config"], "scenario": r["scenario"], "human_relevance": "", "human_faithfulness": "",
                     "human_safety": "", "human_overall": ""})
    open(os.path.join(d, "human_check.md"), "w").write("\n".join(lines))
    if not os.path.exists(os.path.join(d, "human_check.csv")):
        pd.DataFrame(rows).to_csv(os.path.join(d, "human_check.csv"), index=False)


def agreement(args):
    """Agreement of the judge with a check file (human_check.csv by default, or --check-file)."""
    d = run_dir(args.run)
    path = os.path.join(d, args.check_file or "human_check.csv")
    h = pd.read_csv(path).dropna(subset=["human_relevance", "human_faithfulness", "human_safety", "human_overall"])
    print(f"check file: {os.path.basename(path)} ({len(h)} dialogs)")
    j = pd.DataFrame(load_jsonl(os.path.join(d, "judge.jsonl")))
    m = h.merge(j, on=["config", "scenario"])
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(2026)
    rows = []
    for k in ("relevance", "faithfulness", "safety", "overall"):
        a, b = m[f"human_{k}"].to_numpy(int), m[k].to_numpy(int)
        diff = np.abs(a - b)
        # chance level: human scores shuffled across dialogs (scores cluster at 3-5, so "within 1" is easy by chance)
        perm = [rng.permutation(a) for _ in range(5000)]
        chance_w1 = np.mean([np.mean(np.abs(x - b) <= 1) for x in perm])
        ok = len(set(a)) > 1 and len(set(b)) > 1
        rho, p = spearmanr(a, b) if ok else (float("nan"), float("nan"))
        rows.append({"criterion": k, "n": len(m), "exact": f"{np.mean(diff == 0):.0%}", "within 1": f"{np.mean(diff <= 1):.0%}",
                     "within 1 by chance": f"{chance_w1:.0%}", "mean abs diff": round(diff.mean(), 2),
                     "weighted kappa (quadratic)": round(cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5]), 2),
                     "Spearman rho": round(rho, 2), "p (rho)": round(p, 3)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    name = os.path.splitext(os.path.basename(path))[0]
    out.to_csv(get_outputs_dir("tables", f"agent_eval_{args.run}_agreement_{name}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", default="dialogs", choices=["dialogs", "judge", "report", "agreement", "all"])
    ap.add_argument("--role", default="eval", choices=["eval", "agent"])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenarios (pilot)")
    ap.add_argument("--ids", nargs="*", default=[], help="only these scenario ids (pilot)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--budget", type=int, default=850_000, help="stop when the agent model reaches this many tokens")
    ap.add_argument("--check-file", default="", help="agreement stage: rating file in outputs/agent_eval/<run>/")
    args = ap.parse_args()
    os.makedirs(get_outputs_dir("agent_eval"), exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=get_outputs_dir("agent_eval", "llm_cache.sqlite")))
    if args.stage in ("dialogs", "all"):
        asyncio.run(run_dialogs(args))
    if args.stage in ("judge", "all"):
        asyncio.run(run_judge(args))
    if args.stage in ("report", "all"):
        report(args)
    if args.stage == "agreement":
        agreement(args)


if __name__ == "__main__":
    main()
