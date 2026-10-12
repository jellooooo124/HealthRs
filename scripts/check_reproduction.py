"""
Compare a reproduced results directory with the committed one.

    python scripts/check_reproduction.py outputs_reproduced/tables outputs/tables

* every numeric cell of every common CSV is compared, except timing columns (latency, seconds per epoch,
  training minutes), which depend on the machine and its load;
* if both result directories contain the serving checkpoints, their weights must be identical.
"""
import os
import re
import sys

import pandas as pd

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")
TIMING = re.compile(r"latency|sec|train_min|train\+eval", re.I)
SERVING = ("f_bpr_s2026.pt", "f_hA_s2026.pt")


def numbers(df: pd.DataFrame) -> list[float]:
    vals = []
    for v in df.to_numpy().ravel():
        vals += [float(x) for x in NUM.findall(str(v).replace(",", ""))]
    return vals


def compare_tables(new_dir: str, ref_dir: str, tol: float) -> bool:
    ok_all = True
    common = sorted(f for f in set(os.listdir(new_dir)) & set(os.listdir(ref_dir)) if f.endswith(".csv"))
    for f in common:
        a, b = pd.read_csv(os.path.join(new_dir, f)), pd.read_csv(os.path.join(ref_dir, f))
        cols = [c for c in b.columns if c in a.columns and not TIMING.search(str(c))]
        x, y = numbers(a[cols]), numbers(b[cols])
        if len(x) != len(y):
            print(f"  {f:45s} DIFFERENT SHAPE ({len(x)} vs {len(y)} numbers)")
            ok_all = False
            continue
        # a cell matches if |new - ref| <= tol * |ref| (relative), with a tiny absolute floor for zeros
        d = max((abs(p - q) / max(abs(q), 1e-12) for p, q in zip(x, y) if abs(p - q) > 1e-9), default=0.0)
        ok = all(abs(p - q) <= max(tol * abs(q), 1e-9) for p, q in zip(x, y))
        ok_all &= ok
        skipped = len(b.columns) - len(cols)
        print(f"  {f:45s} max relative diff {d:.1e}  {'OK' if ok else 'CHECK'}"
              + (f"  ({skipped} timing column(s) not compared)" if skipped else ""))
    print(f"{len(common)} tables compared")
    return ok_all


def compare_checkpoints(new_dir: str, ref_dir: str) -> bool | None:
    pairs = [(os.path.join(new_dir, n), os.path.join(ref_dir, n)) for n in SERVING]
    if not all(os.path.exists(p) and os.path.exists(q) for p, q in pairs):
        return None
    import torch
    ok_all = True
    for p, q in pairs:
        a = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        b = torch.load(q, map_location="cpu", weights_only=False)["state_dict"]
        ok = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        ok_all &= ok
        print(f"  {os.path.basename(p):45s} {'identical weights' if ok else 'WEIGHTS DIFFER'}")
    return ok_all


def main(new_tables: str, ref_tables: str, tol: float = 1e-3):
    print("Tables:")
    tables_ok = compare_tables(new_tables, ref_tables, tol)
    print("Serving checkpoints:")
    ckpt_ok = compare_checkpoints(os.path.join(os.path.dirname(os.path.normpath(new_tables)), "checkpoints"),
                                  os.path.join(os.path.dirname(os.path.normpath(ref_tables)), "checkpoints"))
    if ckpt_ok is None:
        print("  (not available in both directories)")
    passed = tables_ok and ckpt_ok is not False
    print("REPRODUCTION OK" if passed else "REPRODUCTION: some results differ (see CHECK above)")
    return passed


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:3]) else 1)
