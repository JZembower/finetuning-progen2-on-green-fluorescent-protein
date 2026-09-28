"""Rank AlphaFold 3 predictions by mean pLDDT and flag pTM > 0.8.

Usage:
    python af3_eval/rank_af3_outputs.py af3_work/af_output
    python af3_eval/rank_af3_outputs.py server_downloads/ --index af3_work/af_input/designs_index.csv

Reads either layout, searching the directory recursively:
  * local run_alphafold.py output: <job>/<job>_summary_confidences.json + <job>_confidences.json
  * AlphaFold Server zip, extracted: fold_<job>_summary_confidences_0.json + fold_<job>_full_data_0.json
Only the top-ranked prediction per job is used (local root files / server model 0).

Writes <dir>/af3_ranking.csv and prints the top 5. Standard library only.
"""

import argparse
import csv
import json
import pathlib
import re

SUMMARY_RE = re.compile(r"(?P<job>.+)_summary_confidences(?:_(?P<model>\d+))?\.json")
PTM_THRESHOLD = 0.8


def find_predictions(root):
    for summary in sorted(root.rglob("*summary_confidences*.json")):
        m = SUMMARY_RE.fullmatch(summary.name)
        if not m or "_seed-" in summary.name:
            continue  # Per-sample files; the job's root files already hold the top one.
        job, model = m["job"], m["model"]
        if model is None:
            full = summary.with_name(f"{job}_confidences.json")
        elif model == "0":
            full = summary.with_name(f"{job}_full_data_0.json")
        else:
            continue
        if full.exists():
            yield re.sub(r"^fold_", "", job), summary, full  # Server prefixes job names with fold_.


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--index", type=pathlib.Path, help="designs_index.csv from make_af3_inputs.py")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    sequences = {}
    if args.index:
        with args.index.open() as f:
            sequences = {row["name"].lower(): row["sequence"] for row in csv.DictReader(f)}

    rows = []
    for job, summary_path, full_path in find_predictions(args.output_root):
        summary = json.loads(summary_path.read_text())
        plddts = json.loads(full_path.read_text())["atom_plddts"]
        rows.append({
            "job": job,
            "mean_plddt": round(sum(plddts) / len(plddts), 2),
            "ptm": summary.get("ptm"),
            "ptm_gt_0_8": summary.get("ptm") is not None and summary["ptm"] > PTM_THRESHOLD,
            "ranking_score": summary.get("ranking_score"),
            "fraction_disordered": summary.get("fraction_disordered"),
            "has_clash": summary.get("has_clash"),
            "sequence": sequences.get(job.lower(), ""),
        })

    if not rows:
        raise SystemExit(f"No AlphaFold 3 predictions found under {args.output_root}")

    rows.sort(key=lambda r: r["mean_plddt"], reverse=True)
    out_path = args.output_root / "af3_ranking.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passing = [r for r in rows if r["ptm_gt_0_8"]]
    print(f"{len(rows)} predictions, {len(passing)} with pTM > {PTM_THRESHOLD}. Full table: {out_path}")
    print(f"Top {args.top} by mean pLDDT with pTM > {PTM_THRESHOLD}:")
    for r in passing[: args.top]:
        print(f"  {r['job']:<16} pLDDT={r['mean_plddt']:6.2f}  pTM={r['ptm']:.2f}")


if __name__ == "__main__":
    main()
