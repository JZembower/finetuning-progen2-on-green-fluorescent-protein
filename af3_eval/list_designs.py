#!/usr/bin/env python3
"""List design sequences for submission to the AlphaFold 3 web server.

Design numbers are the 1-based line number in the design file, zero-padded to
three digits, matching make_af3_inputs.py and rank_af3_outputs.py. Use the same
name as the AlphaFold Server job name so the downloaded folder comes back as
fold_design_NNN and the ranking script can find it.

Examples, from the repository root:

    # everything, one per line, as NAME<TAB>SEQUENCE
    python af3_eval/list_designs.py psc_runs/design_topp_10ep.txt

    # just the ones you still need to fold
    python af3_eval/list_designs.py psc_runs/design_topp_10ep.txt \\
        --only 6,11,45,48,73,83

    # everything not already present in a ranking file
    python af3_eval/list_designs.py psc_runs/design_topp_10ep.txt \\
        --unfolded af3_results/af3_ranking.csv

    # only designs the model itself altered beyond the 64-residue prompt
    python af3_eval/list_designs.py psc_runs/design_topp_10ep.txt \\
        --model-authored assignment/data/train.txt

    # a spreadsheet instead
    python af3_eval/list_designs.py psc_runs/design_topp_10ep.txt --csv designs.csv

Standard library only.
"""

import argparse
import csv
import pathlib
import sys

from divergence_analysis import clean, consensus_of, substitutions

PROMPT_LEN = 64


def parse_only(spec):
    """'6,11,45' or '1-20' or a mix of both -> a set of ints."""
    wanted = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(chunk))
    return wanted


def folded_sequences(ranking_path):
    """Sequences already present in a ranking CSV, so they need no refolding."""
    with ranking_path.open() as f:
        return {r["sequence"] for r in csv.DictReader(f) if r.get("sequence")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design_file", type=pathlib.Path)
    ap.add_argument("--prefix", default="design",
                    help="name prefix; must match make_af3_inputs.py (default: design)")
    ap.add_argument("--only", help="design numbers to keep, e.g. 6,11,45 or 1-20")
    ap.add_argument("--unfolded", type=pathlib.Path, metavar="RANKING_CSV",
                    help="keep only designs whose sequence is absent from this ranking")
    ap.add_argument("--model-authored", type=pathlib.Path, metavar="TRAIN_FILE",
                    help="keep only designs with a substitution beyond the prompt")
    ap.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    ap.add_argument("--csv", type=pathlib.Path, help="also write a CSV here")
    ap.add_argument("--fasta", type=pathlib.Path, help="also write FASTA here")
    args = ap.parse_args()

    if not args.design_file.exists():
        sys.exit(f"ERROR: {args.design_file} not found")

    designs = [clean(l) for l in args.design_file.read_text().splitlines() if l.strip()]
    rows = [(f"{args.prefix}_{i:03d}", i, s) for i, s in enumerate(designs, start=1) if s]
    if not rows:
        sys.exit(f"ERROR: no usable sequences in {args.design_file}")
    total = len(rows)

    if args.only:
        wanted = parse_only(args.only)
        rows = [r for r in rows if r[1] in wanted]

    if args.unfolded:
        done = folded_sequences(args.unfolded)
        rows = [r for r in rows if r[2] not in done]

    notes = {}
    if args.model_authored:
        consensus, _, _ = consensus_of(args.model_authored)
        kept = []
        for name, num, seq in rows:
            mine = [f"{c}{p}{d}" for c, p, d in substitutions(consensus, seq)
                    if p > args.prompt_len]
            if mine:
                notes[name] = ", ".join(mine)
                kept.append((name, num, seq))
        rows = kept

    if not rows:
        print("No designs matched the filters.", file=sys.stderr)
        return

    width = max(len(n) for n, _, _ in rows)
    for name, _, seq in rows:
        line = f"{name:<{width}}\t{seq}"
        if name in notes:
            line += f"\t# model-authored: {notes[name]}"
        print(line)

    print(f"\n{len(rows)} of {total} designs listed "
          f"(length {len(rows[0][2])}).", file=sys.stderr)
    print("Use the name as the AlphaFold Server job name, so results download "
          "as fold_<name>/.", file=sys.stderr)

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["design", "number", "length", "sequence", "model_authored"])
            for name, num, seq in rows:
                w.writerow([name, num, len(seq), seq, notes.get(name, "")])
        print(f"wrote {args.csv}", file=sys.stderr)

    if args.fasta:
        args.fasta.write_text("".join(f">{n}\n{s}\n" for n, _, s in rows))
        print(f"wrote {args.fasta}", file=sys.stderr)


if __name__ == "__main__":
    main()
