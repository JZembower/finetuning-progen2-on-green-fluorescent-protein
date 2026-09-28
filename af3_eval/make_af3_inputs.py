"""Turn ProGen2 designs (one sequence per line) into AlphaFold 3 input JSONs.

Usage:
    python af3_eval/make_af3_inputs.py assignment/design/design.txt af3_work/af_input
    python af3_eval/make_af3_inputs.py assignment/design/design.txt af3_work/af_input --with-msa

By default each input is MSA-free (empty MSAs, no templates), so AlphaFold 3 can
run with --norun_data_pipeline and no genetic databases. Pass --with-msa on a
machine that has the databases (e.g. PSC) to let AF3 build MSAs and templates.

Standard library only, so it runs from any Python 3.9+.
"""

import argparse
import csv
import json
import pathlib
import re

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean(seq):
    """Drops ProGen control tokens ('1'/'2' = BOS/EOS) and non-standard residues."""
    seq = re.sub(r"\s+", "", seq).upper()
    kept = "".join(c for c in seq if c in STANDARD_AA)
    return kept, sorted(set(seq) - STANDARD_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("design_file", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--prefix", default="design")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--with-msa",
        action="store_true",
        help="Omit MSA/template fields so AF3's data pipeline builds them.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = [l for l in args.design_file.read_text().splitlines() if l.strip()]

    index_rows = []
    for i, line in enumerate(lines, start=1):
        seq, dropped = clean(line)
        name = f"{args.prefix}_{i:03d}"
        if dropped:
            print(f"{name}: dropped non-standard characters {dropped}")
        if not seq:
            print(f"{name}: empty after cleaning, skipped")
            continue

        protein = {"id": "A", "sequence": seq}
        if not args.with_msa:
            protein.update(unpairedMsa="", pairedMsa="", templates=[])

        fold_input = {
            "name": name,
            "sequences": [{"protein": protein}],
            "modelSeeds": [args.seed],
            "dialect": "alphafold3",
            "version": 4,
        }
        (args.out_dir / f"{name}.json").write_text(json.dumps(fold_input, indent=2))
        index_rows.append({"name": name, "length": len(seq), "sequence": seq})

    index_path = args.out_dir / "designs_index.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "length", "sequence"])
        writer.writeheader()
        writer.writerows(index_rows)

    unique = len({r["sequence"] for r in index_rows})
    mode = "with MSA search" if args.with_msa else "MSA-free"
    print(
        f"Wrote {len(index_rows)} {mode} inputs ({unique} unique sequences) to"
        f" {args.out_dir}; name -> sequence map in {index_path.name}"
    )


if __name__ == "__main__":
    main()
