import argparse
import csv
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    eval_root = args.run_dir / "evaluation"
    paths = {
        "uniform": eval_root / "uniform_full" / "results.json",
        "mmd_selective": eval_root / "mmd_selective_full" / "results.json",
    }
    results = {}
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(p)
        results[name] = json.loads(p.read_text())

    u = results["uniform"]
    m = results["mmd_selective"]
    cfg=json.loads((args.run_dir/'config.json').read_text())
    expected=10 * cfg['evaluation']['full_trials_per_task']
    for name,r in results.items():
        if len(r['per_task']) != 10 or r['total_episodes'] != expected:
            raise RuntimeError(f'{name}: incomplete full evaluation')
        if sorted(x['task_id'] for x in r['per_task']) != list(range(10)):
            raise RuntimeError(f'{name}: missing or duplicate task IDs')
        if not 0 <= r['total_successes'] <= expected: raise RuntimeError('Invalid success count')
        if not math.isclose(r['success_rate'],r['total_successes']/expected): raise RuntimeError('Inconsistent success rate')
    if u['seed'] != m['seed'] or u['env_seed'] != m['env_seed']:
        raise RuntimeError('Conditions use different evaluation seeds')
    training={}
    for condition in results:
        root=args.run_dir/'training'/condition
        completion=json.loads((root/'completion.json').read_text())
        if completion['optimizer_steps'] != cfg['training']['max_steps']: raise RuntimeError('Incomplete training')
        if completion['audit']['trainable_params'] != cfg['expected_trainable_params']: raise RuntimeError('Unequal adapter budget')
        training[condition]={'completion':completion, 'resumed':(root/'resume_disclosure.json').exists()}
    summary = {
        'configuration':cfg,
        'training':training,
        'source_manifest':json.loads((args.run_dir/'source_manifest.json').read_text()),
        'limitations':['Single training seed per condition', 'Offline MMD-selection compute cost not included', 'Resumed RLDS streams are approximate; see training provenance'],
        "uniform_successes": u["total_successes"],
        "uniform_episodes": u["total_episodes"],
        "uniform_success_rate": u["success_rate"],
        "mmd_successes": m["total_successes"],
        "mmd_episodes": m["total_episodes"],
        "mmd_success_rate": m["success_rate"],
        "mmd_minus_uniform_percentage_points":
            100 * (m["success_rate"] - u["success_rate"]),
        "per_task": [],
    }

    for ur, mr in zip(u["per_task"], m["per_task"]):
        if ur["task_id"] != mr["task_id"] or ur['task'] != mr['task']:
            raise RuntimeError("Per-task result ordering mismatch")
        summary["per_task"].append({
            "task_id": ur["task_id"],
            "task": ur["task"],
            "uniform_success_rate": ur["success_rate"],
            "mmd_success_rate": mr["success_rate"],
            "delta_percentage_points":
                100 * (mr["success_rate"] - ur["success_rate"]),
        })

    out_json = args.run_dir / "FINAL_EXPERIMENT_SUMMARY.json"
    out_json.write_text(json.dumps(summary, indent=2))

    out_csv = args.run_dir / "FINAL_PER_TASK_RESULTS.csv"
    with out_csv.open("w", newline="") as f:
        fields = [
            "task_id", "task", "uniform_success_rate",
            "mmd_success_rate", "delta_percentage_points",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary["per_task"])

    print(json.dumps(summary, indent=2))
    print("FINAL SUMMARY:", out_json)
    print("PER-TASK CSV:", out_csv)


if __name__ == "__main__":
    main()
