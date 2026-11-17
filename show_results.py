"""Print a table of every run in runs/, read from the files finetune.py wrote.

Usage (from the project folder):   python show_results.py
"""
import json
from pathlib import Path

PRIMARY = {"regression": "macro_ST-RAE", "classification": "macro_MCC"}
short = lambda t: t.replace("_pIC50_direct_inhibition", "").replace("_is_TDI", "")

for task, key in PRIMARY.items():
    folder = Path("runs") / task
    if not folder.exists():
        print(f"(no runs for {task} yet)\n")
        continue

    better = "lower is better" if task == "regression" else "higher is better"
    print(f"=== {task.upper()}  —  {key}  ({better})")

    rows = []
    for vm in folder.glob("*/*/validation_metrics.json"):
        run = vm.parent
        val = json.loads(vm.read_text())
        score = val.get(key)
        if score is None:
            continue

        # per-endpoint scores
        per = {short(t): m.get("ST-RAE", m.get("MCC"))
               for t, m in val.items() if isinstance(m, dict)}

        # which settings produced this run
        ov = run / ".hydra" / "overrides.yaml"
        settings = " ".join(
            line.strip("- ").strip() for line in ov.read_text().splitlines()
            if line.strip() and not line.strip("- ").startswith(("model=", "task=", "device="))
        ) if ov.exists() else ""

        # best epoch out of how many
        hist, epochs = run / "history.jsonl", ""
        if hist.exists():
            h = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
            scores = [r[key] for r in h]
            best = min(scores) if task == "regression" else max(scores)
            epochs = f"{h[scores.index(best)]['epoch']}/{len(h)}"

        rows.append((score, run.parent.name, run.name, per, epochs, settings))

    rows.sort(reverse=(task == "classification"))
    for score, model, run_name, per, epochs, settings in rows:
        detail = "  ".join(f"{k} {v:.3f}" for k, v in per.items())
        print(f"  {score:.4f}  {model:<18s} {run_name}  best_ep {epochs:>7s}  | {detail}"
              + (f"  | {settings}" if settings else ""))
    print()
