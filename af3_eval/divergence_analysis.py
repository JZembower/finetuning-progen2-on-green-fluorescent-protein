"""Compare designs against the training-set consensus (assignment deliverable 4).

Usage:
    python af3_eval/divergence_analysis.py assignment/data/train.txt assignment/design/design.txt
    python af3_eval/divergence_analysis.py assignment/data/train.txt assignment/design/design.txt \
        --ranking af3_work/af_output/af3_ranking.csv --top 5 --out report_data/

Computes the consensus sequence (most frequent residue per position) and, for each
design, reports every differing position in A36V form (consensus residue, 1-based
position, design residue), plus where along the sequence those substitutions fall.

With --ranking, only the top-N designs by mean pLDDT that also clear pTM > 0.8 are
analysed, in ranked order. Standard library only.

Positions are 1-based to match the A36V convention in the assignment.
"""

import argparse
import collections
import csv
import pathlib
import re

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
N_BINS = 6  # Positional-distribution buckets across the sequence.


def clean(seq):
    """Same cleaning as make_af3_inputs.py, so names and sequences line up."""
    return "".join(c for c in re.sub(r"\s+", "", seq).upper() if c in STANDARD_AA)


def consensus_of(path):
    """Most frequent residue at each position over equal-length training sequences."""
    counts = None
    lengths = collections.Counter()
    for line in path.read_text().splitlines():
        seq = clean(line)
        if not seq:
            continue
        lengths[len(seq)] += 1
        if counts is None:
            counts = [collections.Counter() for _ in range(len(seq))]
        if len(seq) != len(counts):
            continue  # Ignore off-length sequences rather than mis-align columns.
        for i, residue in enumerate(seq):
            counts[i][residue] += 1
    if counts is None:
        raise SystemExit(f"No usable sequences in {path}")

    consensus = "".join(c.most_common(1)[0][0] for c in counts)
    # Conservation = how dominant the top residue is at each position.
    conservation = [c.most_common(1)[0][1] / sum(c.values()) for c in counts]
    return consensus, conservation, lengths


def substitutions(consensus, design):
    """Differing positions as (consensus_residue, 1-based position, design_residue)."""
    return [
        (c, i + 1, d)
        for i, (c, d) in enumerate(zip(consensus, design))
        if c != d
    ]


def load_ranked_names(ranking_path, top):
    """Top-N job names by mean pLDDT among those with pTM > 0.8, best first."""
    with ranking_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("ptm_gt_0_8") == "True"]
    rows.sort(key=lambda r: float(r["mean_plddt"]), reverse=True)
    return [(r["job"], float(r["mean_plddt"]), float(r["ptm"])) for r in rows[:top]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("train_file", type=pathlib.Path)
    parser.add_argument("design_file", type=pathlib.Path)
    parser.add_argument("--ranking", type=pathlib.Path, help="af3_ranking.csv, to pick the top designs")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--prefix", default="design", help="Must match make_af3_inputs.py --prefix")
    parser.add_argument("--out", type=pathlib.Path, help="Directory for CSV output")
    args = parser.parse_args()

    consensus, conservation, lengths = consensus_of(args.train_file)
    print(f"Consensus over {sum(lengths.values())} training sequences, length {len(consensus)}")
    if len(lengths) > 1:
        print(f"  warning: mixed lengths present {dict(lengths)}; used {len(consensus)}")
    mean_cons = sum(conservation) / len(conservation)
    invariant = sum(1 for c in conservation if c == 1.0)
    print(f"  mean per-position conservation {mean_cons:.3f}; {invariant} fully invariant positions")
    print(f"  {consensus}\n")

    designs = [clean(l) for l in args.design_file.read_text().splitlines() if l.strip()]
    named = {f"{args.prefix}_{i:03d}": s for i, s in enumerate(designs, start=1) if s}

    if args.ranking:
        selected = load_ranked_names(args.ranking, args.top)
        missing = [n for n, _, _ in selected if n.lower() not in {k.lower() for k in named}]
        if missing:
            raise SystemExit(f"Ranked designs missing from {args.design_file}: {missing}")
        lookup = {k.lower(): v for k, v in named.items()}
        chosen = [(n, lookup[n.lower()], p, t) for n, p, t in selected]
    else:
        print(f"No --ranking given; analysing the first {args.top} designs in file order.\n")
        chosen = [(n, s, None, None) for n, s in list(named.items())[: args.top]]

    rows = []
    for rank, (name, design, plddt, ptm) in enumerate(chosen, start=1):
        header = f"#{rank} {name}  length {len(design)}"
        if plddt is not None:
            header += f"  pLDDT {plddt:.2f}  pTM {ptm:.2f}"
        print(header)

        if len(design) != len(consensus):
            print(f"  length {len(design)} != consensus {len(consensus)};"
                  " comparing the overlapping prefix only")
        subs = substitutions(consensus, design)
        pct = 100 * len(subs) / min(len(design), len(consensus))
        print(f"  {len(subs)} substitutions ({pct:.1f}% of compared positions)")
        print(f"  {', '.join(f'{c}{p}{d}' for c, p, d in subs) if subs else '  (identical to consensus)'}")

        # Positional distribution across N_BINS equal windows.
        width = len(consensus) / N_BINS
        bins = collections.Counter(min(int((p - 1) / width), N_BINS - 1) for _, p, _ in subs)
        spread = "  ".join(
            f"{int(b*width)+1}-{int((b+1)*width)}:{bins.get(b,0)}" for b in range(N_BINS)
        )
        print(f"  positions: {spread}")

        # Flag substitutions at positions the training set never varies.
        at_invariant = [f"{c}{p}{d}" for c, p, d in subs if conservation[p - 1] == 1.0]
        if at_invariant:
            print(f"  at fully-invariant positions: {', '.join(at_invariant)}")
        print()

        for c, p, d in subs:
            rows.append({
                "rank": rank, "design": name, "mean_plddt": plddt, "ptm": ptm,
                "substitution": f"{c}{p}{d}", "consensus_residue": c,
                "position": p, "design_residue": d,
                "train_conservation_at_position": round(conservation[p - 1], 4),
            })

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "consensus.txt").write_text(consensus + "\n")
        subs_path = args.out / "substitutions.csv"
        with subs_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "rank", "design", "mean_plddt", "ptm", "substitution",
                "consensus_residue", "position", "design_residue",
                "train_conservation_at_position",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {subs_path} ({len(rows)} substitutions) and consensus.txt")


if __name__ == "__main__":
    main()
