#!/usr/bin/env python3
"""Compare design sets from different finetuning runs or decoders.

The question this answers is the one the report turns on: of the substitutions
away from the training consensus, how many did the MODEL actually author?

Positions 1-64 come from the prompt, so a difference there was handed to the
model, not invented by it. Only positions 65-237 are the model's own work.
Counting them together inflates apparent novelty by roughly 60x, which is the
single easiest way to overstate what a run achieved.

Usage, from the repository root:

    python af3_eval/compare_runs.py assignment/data/train.txt \\
        "5ep greedy=assignment/design/design_greedy.txt" \\
        "5ep top-p=assignment/design/design_topp.txt" \\
        "10ep greedy=psc_runs/design_greedy_10ep.txt" \\
        "10ep top-p=psc_runs/design_topp_10ep.txt"

Each argument after the training file is LABEL=PATH. Reuse an existing
consensus with --consensus report_data/consensus.txt to skip recomputing it
from 47k sequences.

Standard library only, like the rest of af3_eval/.
"""

import argparse
import collections
import pathlib
import sys

from divergence_analysis import clean, consensus_of, substitutions

PROMPT_LEN = 64  # test.txt prompts are the first 64 residues.


def load_designs(path):
    seqs = [clean(line) for line in path.read_text().splitlines()]
    return [s for s in seqs if s]


def summarise(consensus, designs, prompt_len):
    """Split each design's substitutions at the prompt boundary."""
    prompt_subs = 0
    model_subs = 0
    designs_with_model_sub = 0
    model_sub_counter = collections.Counter()
    lengths = collections.Counter()

    for seq in designs:
        lengths[len(seq)] += 1
        mine = 0
        for ref, pos, got in substitutions(consensus, seq):
            if pos <= prompt_len:
                prompt_subs += 1
            else:
                model_subs += 1
                mine += 1
                model_sub_counter[f"{ref}{pos}{got}"] += 1
        if mine:
            designs_with_model_sub += 1

    return {
        "n": len(designs),
        "unique": len(set(designs)),
        "lengths": lengths,
        "prompt_subs": prompt_subs,
        "model_subs": model_subs,
        "designs_with_model_sub": designs_with_model_sub,
        "model_sub_counter": model_sub_counter,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train", type=pathlib.Path,
                    help="training sequences, for the consensus")
    ap.add_argument("sets", nargs="+", metavar="LABEL=PATH",
                    help="design files to compare")
    ap.add_argument("--consensus", type=pathlib.Path,
                    help="reuse a consensus.txt instead of recomputing")
    ap.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    args = ap.parse_args()

    if args.consensus and args.consensus.exists():
        consensus = clean(args.consensus.read_text())
        print(f"consensus: {args.consensus} ({len(consensus)} residues)")
    else:
        if not args.train.exists():
            sys.exit(f"ERROR: {args.train} not found")
        consensus, conservation, lengths = consensus_of(args.train)
        print(f"consensus: computed from {args.train} "
              f"({sum(lengths.values())} sequences, {len(consensus)} residues, "
              f"min conservation {min(conservation):.3f})")

    rows = []
    for item in args.sets:
        if "=" not in item:
            sys.exit(f"ERROR: expected LABEL=PATH, got '{item}'")
        label, _, raw = item.partition("=")
        path = pathlib.Path(raw)
        if not path.exists():
            print(f"  SKIP {label}: {path} not found", file=sys.stderr)
            continue
        designs = load_designs(path)
        if not designs:
            print(f"  SKIP {label}: no usable sequences", file=sys.stderr)
            continue
        rows.append((label.strip(), path, summarise(consensus, designs, args.prompt_len)))

    if not rows:
        sys.exit("ERROR: no design sets could be read")

    print(f"\nprompt region = positions 1-{args.prompt_len} (given to the model)")
    print(f"model region  = positions {args.prompt_len + 1}-{len(consensus)} "
          f"(the model's own work)\n")

    header = f"| {'Run':<22} | {'n':>3} | {'uniq':>4} | {'prompt subs':>11} | {'model subs':>10} | {'designs w/ sub':>14} |"
    rule = "|" + "|".join("-" * len(c) for c in header.split("|")[1:-1]) + "|"
    print(header)
    print(rule)
    for label, _, s in rows:
        print(f"| {label:<22} | {s['n']:>3} | {s['unique']:>4} | "
              f"{s['prompt_subs']:>11} | {s['model_subs']:>10} | "
              f"{str(s['designs_with_model_sub']) + '/' + str(s['n']):>14} |")

    print("\nModel-authored substitutions (position > "
          f"{args.prompt_len}), in the report's A36V form:")
    for label, _, s in rows:
        if s["model_sub_counter"]:
            listed = ", ".join(f"{k} (x{v})" if v > 1 else k
                               for k, v in sorted(s["model_sub_counter"].items(),
                                                  key=lambda kv: int(kv[0][1:-1])))
            print(f"  {label}: {listed}")
        else:
            print(f"  {label}: none - every design reproduced the consensus "
                  f"beyond position {args.prompt_len}")

    odd = [(label, dict(s["lengths"])) for label, _, s in rows
           if set(s["lengths"]) != {len(consensus)}]
    if odd:
        print("\nWARNING: design lengths differ from the consensus "
              f"({len(consensus)}); substitution positions may be misaligned:")
        for label, lengths in odd:
            print(f"  {label}: {lengths}")


if __name__ == "__main__":
    main()
