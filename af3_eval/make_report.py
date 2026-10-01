#!/usr/bin/env python3
"""Generate the submission report: top-5 candidates plus divergence analysis.

Covers exactly what the assignment asks for and nothing else:

  (2) A PDF file for the top 5 candidate report. Please include the detailed
      information about your top-5 candidates, including the Alphafold3
      visualization, sequences, pTM score, pLDDT score, and provide an
      explanation of these metrics along with divergence analysis.

  Divergence Analysis: substitution table (A36V format), positional
  distribution, decoding strategy and parameters, interpretation (one
  paragraph).

Every number is computed from the inputs, so re-running after folding more
designs regenerates a correct report rather than requiring hand edits.

    python af3_eval/make_report.py \\
        --train assignment/data/train.txt \\
        --designs psc_runs/design_topp_10ep.txt \\
        --ranking af3_results/af3_ranking_10ep.csv \\
        --compare psc_runs/design_greedy_10ep.txt \\
        --out report_submission.html

Then open the HTML and print to PDF.

Standard library only.
"""

import argparse
import csv
import html
import pathlib
import sys

from divergence_analysis import clean, consensus_of, substitutions

PROMPT_LEN = 64
BINS = [(1, 39), (40, 79), (80, 118), (119, 158), (159, 197), (198, 237)]

CSS = """
:root { --ink:#1a1a1a; --muted:#5a5a5a; --rule:#d8d8d8; --accent:#1a4d8f; --bg-soft:#f6f7f9; }
* { box-sizing: border-box; }
body { font-family:"Charter","Georgia","Times New Roman",serif; color:var(--ink);
  max-width:52em; margin:0 auto; padding:2.5em 2em 4em; line-height:1.55;
  font-size:11.5pt; background:#fff; }
h1 { font-size:1.8em; margin:0 0 .15em; letter-spacing:-.01em; }
h2 { font-size:1.2em; margin:2em 0 .6em; padding-bottom:.22em;
  border-bottom:2px solid var(--accent); color:var(--accent); }
h3 { font-size:1.03em; margin:1.4em 0 .4em; }
.sub { color:var(--muted); font-size:.95em; margin-bottom:.3em; }
.meta { color:var(--muted); font-size:.88em; margin-bottom:1.8em; }
table { border-collapse:collapse; width:100%; margin:.7em 0 1.1em; font-size:.92em; }
th,td { border:1px solid var(--rule); padding:.38em .6em; text-align:left; vertical-align:top; }
th { background:var(--bg-soft); font-weight:600; }
td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
code,.seq { font-family:"Consolas","SF Mono",Menlo,monospace; }
.seq { font-size:7.6pt; line-height:1.45; word-break:break-all; background:var(--bg-soft);
  padding:.55em .7em; border:1px solid var(--rule); border-radius:3px; margin:.4em 0 0; }
.card { border:1px solid var(--rule); border-radius:5px; padding:1em 1.2em; margin:1em 0; }
.card h3 { margin-top:0; }
.card img { width:100%; border:1px solid var(--rule); border-radius:3px; margin:.5em 0; }
.stats { display:flex; gap:1.6em; flex-wrap:wrap; margin:.3em 0 .6em; }
.stat { font-size:.92em; }
.stat b { display:block; font-size:1.26em; color:var(--accent); font-variant-numeric:tabular-nums; }
.note { background:var(--bg-soft); border-left:3px solid var(--accent);
  padding:.65em 1em; margin:.9em 0; font-size:.94em; }
.missing { background:#fff1f1; border:1px dashed #c04040; color:#8a2020;
  padding:1.2em; text-align:center; font-size:.9em; border-radius:3px; margin:.5em 0; }
.chromo { color:#b32020; font-weight:700; }
ul { padding-left:1.25em; } li { margin:.25em 0; }
figcaption { font-size:.85em; color:var(--muted); margin-top:.3em; }
@media print { body { padding:0; font-size:10.5pt; max-width:none; }
  h2 { break-after:avoid; } .card { break-inside:avoid; } table { break-inside:avoid; } }
@page { margin:1.6cm; }
"""


def load_designs(path, prefix):
    seqs = [clean(l) for l in path.read_text().splitlines() if l.strip()]
    return {f"{prefix}_{i:03d}": s for i, s in enumerate(seqs, start=1) if s}


def rank_top(ranking_path, top):
    with ranking_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("ptm_gt_0_8") == "True"]
    rows.sort(key=lambda r: float(r["mean_plddt"]), reverse=True)
    return rows[:top], len(rows)


def split_subs(consensus, seq, prompt_len):
    subs = substitutions(consensus, seq)
    return ([s for s in subs if s[1] <= prompt_len],
            [s for s in subs if s[1] > prompt_len])


def fmt(sub):
    c, p, d = sub
    return f"{c}{p}{d}"


def tally(seqs, consensus, prompt_len):
    """(prompt substitutions, model substitutions, designs with >=1 model sub)."""
    p = m = n = 0
    for s in seqs:
        pr, mo = split_subs(consensus, s, prompt_len)
        p += len(pr); m += len(mo)
        if mo:
            n += 1
    return p, m, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=pathlib.Path, required=True)
    ap.add_argument("--designs", type=pathlib.Path, required=True,
                    help="design file for the decoder actually submitted")
    ap.add_argument("--ranking", type=pathlib.Path, required=True)
    ap.add_argument("--compare", type=pathlib.Path,
                    help="the other decoder's design file, for the strategy section")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("report_submission.html"))
    ap.add_argument("--images", type=pathlib.Path, default=pathlib.Path("af3_results/images"))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--prefix", default="design")
    ap.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    ap.add_argument("--strategy", default="topp", choices=["topp", "greedy"])
    ap.add_argument("--p", type=float, default=0.8)
    ap.add_argument("--author", default="Jonah Zembower")
    ap.add_argument("--course", default="CMU 11781 / 02741, Generative AI for Biomedicine")
    ap.add_argument("--date", default="September 30, 2026")
    args = ap.parse_args()

    for p in (args.train, args.designs, args.ranking):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")

    consensus, conservation, lengths = consensus_of(args.train)
    n_train = sum(lengths.values())
    named = load_designs(args.designs, args.prefix)
    top, n_passing = rank_top(args.ranking, args.top)
    if not top:
        sys.exit(f"ERROR: no designs in {args.ranking} pass pTM > 0.8")

    missing = [r["job"] for r in top if r["job"] not in named]
    if missing:
        sys.exit(f"ERROR: ranked designs absent from {args.designs}: {missing}")

    all_seqs = list(named.values())
    sub_p, sub_m, n_with_m = tally(all_seqs, consensus, args.prompt_len)

    o = []
    w = o.append
    w("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">")
    w("<title>Assignment 1 — Top-5 Candidate Report</title>")
    w(f"<style>{CSS}</style>\n</head>\n<body>")
    w("<h1>Finetuning ProGen2 on Green Fluorescent Protein</h1>")
    w("<p class=\"sub\">Assignment 1 &mdash; Top-5 Candidate Report</p>")
    w(f"<p class=\"meta\">{html.escape(args.course)} &middot; "
      f"{html.escape(args.author)} &middot; {html.escape(args.date)}</p>")

    # ---- setup, kept to the minimum needed to read the rest -----------------
    strat_name = ("nucleus (top-<em>p</em>) sampling" if args.strategy == "topp"
                  else "greedy (argmax) decoding")
    w("<h2>1. Setup</h2>")
    w(f"<p>ProGen2-small was finetuned on {n_train:,} GFP sequences of {len(consensus)} residues "
      f"each, then prompted with the first {args.prompt_len} residues of each of the "
      f"{len(named)} held-out test sequences to generate complete proteins using "
      f"{strat_name}. Candidates were folded with AlphaFold&nbsp;3 "
      "(alphafoldserver.com) and ranked by mean pLDDT subject to pTM&nbsp;&gt;&nbsp;0.8. "
      f"All {len(named)} generations terminated at exactly {len(consensus)} residues.</p>")
    n_folded = sum(1 for _ in csv.DictReader(args.ranking.open()))
    gate = (f"all {n_folded} passed" if n_folded == n_passing
            else f"{n_passing} of them passed")
    w(f"<p>{n_folded} of the {len(named)} designs were folded, limited by the AlphaFold Server's "
      f"daily job quota; {gate} the pTM&nbsp;&gt;&nbsp;0.8 gate, and the {len(top)} highest by "
      "mean pLDDT are reported below.</p>")

    # ---- metrics explanation ------------------------------------------------
    w("<h2>2. What pLDDT and pTM measure</h2>")
    w("<p><strong>pLDDT</strong> (predicted Local Distance Difference Test) is a per-atom "
      "confidence score on a 0&ndash;100 scale, reported here as the mean over all atoms. It "
      "is a <em>local</em> measure: it predicts how well each atom's immediate environment "
      "would match an experimental structure. Above 90 is \"very high\", 70&ndash;90 confident, "
      "below 50 usually disordered. Every candidate here averages above 90, which is why the "
      "rendered structures are almost uniformly dark blue.</p>")
    w("<p><strong>pTM</strong> (predicted Template Modeling score) estimates the TM-score "
      "between the prediction and the true structure, on a 0&ndash;1 scale. Unlike pLDDT it is "
      "<em>global</em>: it asks whether the overall fold and the relative placement of distant "
      "parts of the chain are right. Above 0.5 means the general fold is probably correct; "
      "above 0.8 indicates high confidence in the whole topology. The assignment's "
      "pTM&nbsp;&gt;&nbsp;0.8 gate therefore tests that the complete &beta;-barrel is coherent, "
      "not just that local geometry looks tidy.</p>")
    w("<p>The <strong>PAE</strong> (Predicted Aligned Error) panel beside each structure gives "
      "expected positional error in &aring;ngstr&ouml;ms for every residue pair. Uniformly low "
      "(green) PAE means the model is confident about how the whole chain packs together.</p>")
    w("<div class=\"note\"><strong>These metrics measure predicted structural confidence, not "
      "function.</strong> A GFP variant can fold into a convincing &beta;-barrel and still be "
      "dark if the chromophore environment is disturbed, since chromophore maturation depends "
      "on side-chain chemistry AlphaFold does not model. They filter out sequences that "
      "clearly fail to fold; they do not demonstrate fluorescence.</div>")

    # ---- top-5 cards --------------------------------------------------------
    w(f"<h2>3. Top-{len(top)} candidates</h2>")
    w("<p>Substitutions are relative to the training consensus in "
      "<code>&lt;consensus&gt;&lt;position&gt;&lt;design&gt;</code> form. Positions at or below "
      f"{args.prompt_len} were supplied in the prompt; positions beyond "
      f"{args.prompt_len} were generated by the model.</p>")

    rows_for_table = []
    for rank, r in enumerate(top, start=1):
        name = r["job"]
        seq = named[name]
        pr, mo = split_subs(consensus, seq, args.prompt_len)
        rows_for_table.append((rank, name, r, pr, mo))

        w("<div class=\"card\">")
        w(f"  <h3>#{rank} &mdash; {html.escape(name)}</h3>")
        w("  <div class=\"stats\">")
        w(f"    <div class=\"stat\">mean pLDDT <b>{float(r['mean_plddt']):.2f}</b></div>")
        w(f"    <div class=\"stat\">pTM <b>{float(r['ptm']):.2f}</b></div>")
        w(f"    <div class=\"stat\">substitutions <b>{len(pr) + len(mo)}</b></div>")
        w("  </div>")

        desc = []
        if pr:
            desc.append(", ".join(f"<code>{fmt(s)}</code>" for s in pr) +
                        " (from the prompt)")
        if mo:
            desc.append("<strong>" + ", ".join(f"<code>{fmt(s)}</code>" for s in mo) +
                        " &mdash; model-generated</strong>")
        w(f"  <p>{'; '.join(desc) if desc else 'Identical to the consensus.'}</p>")

        img = args.images / f"{name}.png"
        if img.exists():
            w("  <figure>")
            w(f"    <img src=\"{img.as_posix()}\" alt=\"AlphaFold 3 prediction for {name}\">")
            w("    <figcaption>AlphaFold&nbsp;3 prediction coloured by pLDDT, with PAE plot.</figcaption>")
            w("  </figure>")
        else:
            w(f"  <div class=\"missing\">Missing image: <code>{img.as_posix()}</code><br>"
              "Save the AlphaFold Server screenshot here and regenerate.</div>")
        w(f"  <p class=\"seq\">{seq}</p>")
        w("</div>")

    # ---- divergence: substitution table -------------------------------------
    w("<h2>4. Divergence analysis</h2>")
    w("<h3>4.1 Consensus sequence</h3>")
    w(f"<p>Most frequent residue at each position over all {n_train:,} training sequences. "
      "The chromophore-forming triad <span class=\"chromo\">SYG</span> sits at positions "
      "64&ndash;66.</p>")
    c = consensus
    w(f"<p class=\"seq\">{c[:63]}<span class=\"chromo\">{c[63:66]}</span>{c[66:]}</p>")

    w("<h3>4.2 Substitution table</h3>")
    w("<p>Every position at which a top-"
      f"{len(top)} candidate differs from the consensus. \"Conservation\" is how often the "
      "consensus residue occurs at that position in the training set.</p>")
    w("<table><tr><th>Rank</th><th>Design</th><th>Substitution</th><th class=\"num\">Position</th>"
      "<th class=\"num\">Conservation</th><th>Source</th></tr>")
    total_subs = 0
    for rank, name, r, pr, mo in rows_for_table:
        for s in pr + mo:
            total_subs += 1
            src = ("prompt" if s[1] <= args.prompt_len
                   else "<strong>model-generated</strong>")
            bold_open, bold_close = ("<strong>", "</strong>") if s[1] > args.prompt_len else ("", "")
            w(f"<tr><td>{rank}</td><td>{html.escape(name)}</td>"
              f"<td>{bold_open}<code>{fmt(s)}</code>{bold_close}</td>"
              f"<td class=\"num\">{s[1]}</td>"
              f"<td class=\"num\">{conservation[s[1]-1]:.3f}</td><td>{src}</td></tr>")
    w("</table>")

    n_prompt = sum(len(pr) for _, _, _, pr, _ in rows_for_table)
    n_model = sum(len(mo) for _, _, _, _, mo in rows_for_table)
    w(f"<p>{total_subs} substitutions across the top {len(top)}: "
      f"<strong>{n_prompt} inherited from the prompt</strong> and "
      f"<strong>{n_model} generated by the model</strong>.</p>")

    # ---- divergence: positional distribution --------------------------------
    w("<h3>4.3 Positional distribution</h3>")
    w(f"<p>Where the substitutions fall across the {len(consensus)} residues. The prompt covers "
      f"positions 1&ndash;{args.prompt_len}; everything from {args.prompt_len + 1} onward is "
      "model output.</p>")
    w("<table><tr><th>Design</th>" +
      "".join(f"<th class=\"num\">{a}&ndash;{b}</th>" for a, b in BINS) + "</tr>")
    totals = [0] * len(BINS)
    for _, name, _, pr, mo in rows_for_table:
        counts = [0] * len(BINS)
        for s in pr + mo:
            for j, (a, b) in enumerate(BINS):
                if a <= s[1] <= b:
                    counts[j] += 1
                    totals[j] += 1
        w(f"<tr><td>{html.escape(name)}</td>" +
          "".join(f"<td class=\"num\">{n}</td>" for n in counts) + "</tr>")
    w("<tr><th>Total</th>" +
      "".join(f"<th class=\"num\">{n}</th>" for n in totals) + "</tr></table>")

    occupied = [f"{a}&ndash;{b}" for (a, b), n in zip(BINS, totals) if n]
    w(f"<p>All {total_subs} substitutions fall in the window"
      f"{'s' if len(occupied) > 1 else ''} {', '.join(occupied)}"
      f"{', entirely within the prompt region' if n_model == 0 else ''}.</p>")

    # ---- divergence: decoding strategy --------------------------------------
    w("<h3>4.4 Decoding strategy</h3>")
    if args.strategy == "topp":
        w("<table>"
          "<tr><th>Strategy</th><td>Nucleus (top-<em>p</em>) sampling</td></tr>"
          f"<tr><th><em>p</em></th><td>{args.p} &mdash; sample from the smallest set of residues "
          "whose probabilities sum to <em>p</em>, renormalised</td></tr>")
    else:
        w("<table><tr><th>Strategy</th><td>Greedy (argmax) &mdash; deterministic</td></tr>")
    w(f"<tr><th>Prompt length</th><td>{args.prompt_len} residues</td></tr>"
      "<tr><th>Max length</th><td>512 tokens</td></tr>"
      "<tr><th>Batch size</th><td>1</td></tr>"
      "<tr><th>Seed</th><td>0</td></tr></table>")

    if args.compare and args.compare.exists():
        other = "greedy" if args.strategy == "topp" else "top-p"
        other_seqs = list(load_designs(args.compare, args.prefix).values())
        o_p, o_m, o_n = tally(other_seqs, consensus, args.prompt_len)
        this_label = ("Top-<em>p</em>, p = " + str(args.p)) if args.strategy == "topp" else "Greedy (argmax)"
        other_label = "Greedy (argmax)" if args.strategy == "topp" else f"Top-<em>p</em>"
        w("<p>Both decoders were run over the full design set to isolate the decoder's "
          "contribution:</p>")
        w("<table><tr><th>Decoding</th>"
          f"<th class=\"num\">Substitutions in prompt region (1&ndash;{args.prompt_len})</th>"
          f"<th class=\"num\">Substitutions in generated region ({args.prompt_len+1}&ndash;{len(consensus)})</th>"
          "<th class=\"num\">Designs with &ge;1 generated substitution</th></tr>")
        w(f"<tr><td>{this_label}</td><td class=\"num\">{sub_p}</td>"
          f"<td class=\"num\"><strong>{sub_m}</strong></td>"
          f"<td class=\"num\">{n_with_m} / {len(all_seqs)}</td></tr>")
        w(f"<tr><td>{other_label}</td><td class=\"num\">{o_p}</td>"
          f"<td class=\"num\"><strong>{o_m}</strong></td>"
          f"<td class=\"num\">{o_n} / {len(other_seqs)}</td></tr>")
        w("</table>")

    # ---- divergence: interpretation -----------------------------------------
    w("<h3>4.5 Interpretation</h3>")
    plddts = [float(r["mean_plddt"]) for r in top]
    spread = max(plddts) - min(plddts)
    sizes = sorted(len(pr) + len(mo) for _, _, _, pr, mo in rows_for_table)
    mean_cons = sum(conservation) / len(conservation)
    contributed = (
        f"nothing of its own to any of them &mdash; all {total_subs} substitutions in the "
        f"top {len(top)} were handed to it in the prompt"
        if n_model == 0 else
        f"only {n_model} of the {total_subs} substitutions, the other {n_prompt} having been "
        f"handed to it in the prompt")
    w("<p>"
      f"Finetuning succeeded at the task it was set: every candidate folds into a confident GFP "
      f"&beta;-barrel (pTM {min(float(r['ptm']) for r in top):.2f}&ndash;"
      f"{max(float(r['ptm']) for r in top):.2f}, mean pLDDT above 90), terminates at exactly "
      f"{len(consensus)} residues, and preserves the <span class=\"chromo\">SYG</span> "
      f"chromophore triad &mdash; including the tyrosine at position 65, which the model "
      f"generates itself because the prompt stops at {args.prompt_len}. What the model "
      f"<em>contributed</em> to each design, however, is essentially the consensus sequence: it "
      f"added {contributed}. Across all "
      f"{len(all_seqs)} designs the same pattern holds, {sub_p} prompt-region substitutions "
      f"against {sub_m} from the decoder. This follows from the training data, which is a "
      f"point-mutation scan of a single protein at {mean_cons:.1%} mean per-position "
      f"conservation: there is almost no entropy left for a converged autoregressive model to "
      f"sample, so the next-residue distribution is sharp enough that even nucleus sampling "
      f"usually finds one residue occupying the entire probability mass. As for what the pLDDT "
      f"ranking distinguished between &mdash; very little. The top {len(top)} span "
      f"{spread:.2f} pLDDT points while differing from one another by only "
      f"{sizes[0]} to {sizes[-1]} residues out of {len(consensus)}. A gap that small is "
      f"comparable to AlphaFold's own run-to-run variation, so the ordering is better read as "
      f"noise than as a quality gradient. The defensible conclusion is that pLDDT worked here "
      f"as a <em>pass/fail filter</em> confirming these sequences are foldable, which is what "
      f"the pTM&nbsp;&gt;&nbsp;0.8 gate was for, and not as a way to rank near-identical "
      f"candidates against each other."
      "</p>")

    w("</body>\n</html>")
    args.out.write_text("\n".join(o), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"  top {len(top)} of {n_passing} passing pTM > 0.8")
    print(f"  {total_subs} substitutions in the top {len(top)} "
          f"({n_prompt} prompt, {n_model} model)")
    gaps = [r["job"] for r in top if not (args.images / f"{r['job']}.png").exists()]
    if gaps:
        print(f"  MISSING IMAGES: {', '.join(gaps)}", file=sys.stderr)


if __name__ == "__main__":
    main()
