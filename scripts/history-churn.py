#!/usr/bin/env python3
"""Measure corrective rework in the fork's history, indexed by work done rather than by time.

For every fork commit C, each source line that C deletes or rewrites is blamed (restricted to the
fork range) to find the fork commit O that wrote it. If O is one of the K fork commits preceding C,
C *redoes* those lines of O. Whitespace-only changes and blank lines are ignored.

Calendar time is deliberately not used as the x-axis: agent sessions happened whenever a human had
time, so wall-clock gaps say nothing about the amount of work. Two work-based clocks are used:
  * fork commit number (topological order of the current lineage), and
  * cumulative new source lines written (an "effort clock" that is robust to commit granularity).

Measures (all "recent" = written at most K fork commits earlier):
  * follow-ups of a change   number of distinct later commits that redo >= 2 of its lines
  * corrective commit        a commit that redoes >= MIN_LINES recent lines, and those are
                             >= MIN_SHARE of the lines it adds (i.e. its main job is redoing)
  * line ratio               recent lines redone / new lines written (volume of wasted effort)

Follow-ups look forward, so the last K commits are right-censored (biased low).

Scope: src/** and packages/*/src/**, excluding *.test.ts. Upstream-authored commits are excluded
from the x-axis and their lines never count as redone. Pass --worktree to append the uncommitted
working tree as a final pseudo-commit "WT" (off by default so the figure is reproducible from git).

Outputs (in --out, default remediation/):
  history-churn.csv         one row per fork commit
  history-churn-edges.csv   blame edges: commit fixer_n redid `lines` lines written by origin_n
  history-churn.png         the figure
Requires matplotlib + numpy (pip install matplotlib).

Usage: python3 scripts/history-churn.py [--k 20] [--window 40] [--bin 1500] [--worktree]
                                        [--since REV]   # review rounds: summarize REV..HEAD
"""

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
from collections import defaultdict

SRC = re.compile(r"^(src/|packages/[^/]+/src/)")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
BLAME_HDR = re.compile(r"^([0-9a-f]{40}) \d+ \d+(?: \d+)?$")
FORK_AUTHOR = re.compile(r"^Olli")

# (commit, label) markers drawn on every panel.
EVENTS = [
    ("3c3dc43", "non-consuming\nownership model"),
    ("1f74630", "agent milestone\ntask boards"),
    ("25cd207", "AEP gate"),
    ("9ef4083", "block_map\nphases"),
    ("81c0b10", "WIP merge"),
    ("9217ea0", "wasm-multi-\nthreading"),
]
# Large tear-downs that dominate the line ratio: (commit, label, text x offset, text y).
PIVOTS = [
    ("3c3dc43", "autoRef machinery from the\nprevious day replaced", 20, 0.55),
    ("d3777e8", "peephole optimizer added\nand removed the same day", 30, 1.05),
    ("db6a870", "3.5k-line Phase 5b shader path\ndeleted 6 days after it landed", 25, 0.75),
]
# Notes placed inside effort-clock bins that would otherwise look like missing data.
EFFORT_NOTES = [
    (
        "f301725",
        "Phase 5b: 3.5k new\nlines, no repairs —\n~95% deleted\nwholesale 6 days\nlater (see middle)",
    ),
]


def git(*args: str) -> str:
    return subprocess.check_output(("git",) + args, stderr=subprocess.DEVNULL).decode(
        errors="ignore"
    )


def is_src(path: str) -> bool:
    return bool(SRC.match(path)) and not path.endswith(".test.ts")


def diff_stats(diff: str):
    """Return (non-blank added lines, removed lines, {path: [(start, end)] old-side ranges})."""
    added = removed = 0
    ranges = defaultdict(list)
    cur = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            cur = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("+++ "):
            cur = cur or (line[6:] if line.startswith("+++ b/") else None)
            if cur and not is_src(cur):
                cur = None
        elif cur is None:
            continue
        elif line.startswith("@@"):
            m = HUNK.match(line)
            start, count = int(m.group(1)), int(m.group(2) or 1)
            if count > 0:
                ranges[cur].append((start, start + count - 1))
                removed += count
        elif line.startswith("+") and line[1:].strip():
            added += 1
    return added, removed, ranges


def blame_origins(base: str, rev: str, path: str, ranges) -> dict:
    """Count non-blank old-side lines per originating commit (fork range only)."""
    args = ["blame", "-w", "--porcelain"]
    for a, b in ranges:
        args += ["-L", f"{a},{b}"]
    args += [f"{base}..{rev}", "--", path]
    try:
        out = git(*args)
    except subprocess.CalledProcessError:
        return {}
    counts = defaultdict(int)
    sha = None
    for line in out.splitlines():
        m = BLAME_HDR.match(line)
        if m:
            sha = m.group(1)
        elif line.startswith("\t") and sha and line[1:].strip():
            counts[sha] += 1
    return counts


def collect(include_worktree: bool):
    revs = git("rev-list", "--reverse", "--topo-order", "--no-merges", "HEAD").split()
    meta = {}
    for line in git("log", "--no-merges", "--format=%H%x00%at%x00%an%x00%s", "HEAD").splitlines():
        h, ts, an, s = line.split("\x00", 3)
        meta[h] = dict(ts=int(ts), author=an, subject=s)
    first_fork = next(r for r in revs if FORK_AUTHOR.match(meta[r]["author"]))
    base = git("rev-parse", first_fork + "^").strip()
    fork = [r for r in revs[revs.index(first_fork) :] if FORK_AUTHOR.match(meta[r]["author"])]
    fidx = {r: i for i, r in enumerate(fork)}

    def record(rev_old, rev_new_args, i):
        added, removed, ranges = diff_stats(git("diff", "-w", "-U0", "--no-renames", *rev_new_args))
        for path, rs in ranges.items():
            for o, n in blame_origins(base, rev_old, path, rs).items():
                if o in fidx and fidx[o] < i:
                    edges.append((i, fidx[o], n))
        return added, removed

    rows, edges = [], []
    for i, c in enumerate(fork):
        added, removed = record(c + "^", (c + "^", c), i)
        m = meta[c]
        rows.append(dict(sha=c[:7], ts=m["ts"], subject=m["subject"], added=added, removed=removed))
        if i % 100 == 0:
            print(f"  {i}/{len(fork)} commits", flush=True)

    if include_worktree:
        i = len(fork)
        added, removed = record("HEAD", ("HEAD",), i)
        for path in git("ls-files", "--others", "--exclude-standard").split():
            if is_src(path):
                with open(path, errors="ignore") as f:
                    added += sum(1 for line in f if line.strip())
        if added or removed:
            rows.append(
                dict(
                    sha="WT",
                    ts=int(dt.datetime.now().timestamp()),
                    subject="(uncommitted working tree)",
                    added=added,
                    removed=removed,
                )
            )
    return rows, edges


def measures(rows, edges, k, min_lines, min_share):
    n = len(rows)
    redone_by = [0] * n  # recent lines redone by commit i
    redone_of = [0] * n  # lines of commit i redone within the next k commits
    fixers = defaultdict(set)
    for c, o, lines in edges:
        if 0 < c - o <= k:
            redone_by[c] += lines
            redone_of[o] += lines
            if lines >= 2:
                fixers[o].add(c)
    followups = [len(fixers[i]) for i in range(n)]
    corrective = [
        int(redone_by[i] >= min_lines and redone_by[i] >= min_share * max(rows[i]["added"], 1))
        for i in range(n)
    ]
    return redone_by, redone_of, followups, corrective


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, default=20, help="recency horizon in fork commits")
    ap.add_argument("--window", type=int, default=40, help="rolling window in fork commits")
    ap.add_argument("--bin", type=int, default=1500, help="effort-clock bin size in new lines")
    ap.add_argument("--min-lines", type=int, default=3, help="corrective commit: min lines redone")
    ap.add_argument("--min-share", type=float, default=0.3, help="corrective commit: redone/added")
    ap.add_argument("--worktree", action="store_true", help="append uncommitted working tree")
    ap.add_argument(
        "--since", metavar="REV", help="also summarize fork commits in REV..HEAD (review rounds)"
    )
    ap.add_argument("--out", default="remediation")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    print("collecting blame data …")
    rows, edges = collect(args.worktree)
    k = args.k
    redone_by, redone_of, followups, corrective = measures(
        rows, edges, k, args.min_lines, args.min_share
    )
    redone_by5, _, _, _ = measures(rows, edges, 5, args.min_lines, args.min_share)
    redone_by100, _, _, _ = measures(rows, edges, 100, args.min_lines, args.min_share)
    has_wt = rows[-1]["sha"] == "WT"
    nc = len(rows) - 1 if has_wt else len(rows)  # committed history only below

    # First fork commit inside the last third of *all* repository commits (incl. upstream).
    order = git("rev-list", "--reverse", "--topo-order", "HEAD").split()
    fork_short = {r["sha"]: i + 1 for i, r in enumerate(rows)}
    third_n = next(
        (fork_short[h[:7]] for h in order[len(order) * 2 // 3 :] if h[:7] in fork_short), None
    )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "history-churn-edges.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fixer_n", "origin_n", "lines"])
        w.writerows([c + 1, o + 1, lines] for c, o, lines in sorted(edges))
    with open(os.path.join(args.out, "history-churn.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "sha",
                "date",
                "added",
                "removed",
                f"redone_by_k{k}",
                f"redone_of_k{k}",
                f"followups_k{k}",
                "corrective",
                "subject",
            ]
        )
        for i, r in enumerate(rows):
            date = dt.datetime.fromtimestamp(r["ts"], dt.timezone.utc).strftime("%Y-%m-%d")
            w.writerow(
                [
                    i + 1,
                    r["sha"],
                    date,
                    r["added"],
                    r["removed"],
                    redone_by[i],
                    redone_of[i],
                    followups[i],
                    corrective[i],
                    r["subject"],
                ]
            )

    added = np.array([r["added"] for r in rows[:nc]], float)
    code = np.array([1.0 if r["added"] or r["removed"] else 0.0 for r in rows[:nc]])
    ones = np.ones(args.window)

    def roll(num, den):
        a = np.convolve(np.asarray(num[:nc], float), ones, "same")
        b = np.convolve(np.asarray(den[:nc], float), ones, "same")
        return np.where(b > 0, a / np.maximum(b, 1e-9), np.nan)

    x = np.arange(1, nc + 1)
    cum = np.cumsum(added) / 1000
    sha_to_n = {r["sha"]: i + 1 for i, r in enumerate(rows)}

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 13), gridspec_kw=dict(height_ratios=[3, 2.2, 2.2], hspace=0.5)
    )

    # Panel 1: fix-on-fix, counted in commits.
    ax1.plot(
        x,
        roll(np.array(followups[:nc]) * code, code),
        color="#2c6fbb",
        lw=2,
        label=f"follow-ups per change: later commits (≤{k} commits on) that redo its code",
    )
    ax1.plot(
        x,
        roll(np.array(corrective[:nc]) * code, code),
        color="#c0392b",
        lw=2,
        label="corrective-commit share: commits whose main job is redoing recent code",
    )
    ax1.axvspan(nc - k, nc, color="#2c6fbb", alpha=0.08, lw=0)
    ax1.text(
        nc - k / 2,
        0.03,
        "follow-ups\ncensored\n(biased low)",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#2c6fbb",
        transform=ax1.get_xaxis_transform(),
    )
    ax1.set_ylabel("per code-touching commit\n(rolling mean)")
    ax1.set_title(
        f"Fix-on-fix: how often changes need follow-up repairs "
        f"(rolling window of {args.window} commits)",
        fontsize=11,
    )
    ax1.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.79), framealpha=0.92)

    # Panel 2: volume of wasted effort.
    ax2.plot(
        x,
        roll(redone_by100[:nc], added),
        color="#7d3c98",
        lw=1.2,
        ls="--",
        label="same, recent = ≤100 commits old",
    )
    ax2.plot(
        x,
        roll(redone_by[:nc], added),
        color="#7d3c98",
        lw=2,
        label=f"recent lines redone / new lines (recent = ≤{k} commits old)",
    )
    ax2.plot(
        x,
        roll(redone_by5[:nc], added),
        color="#c39bd3",
        lw=1.3,
        label="same, recent = ≤5 commits old",
    )
    ratio100 = roll(redone_by100[:nc], added)
    for sha, text, dx, ty in PIVOTS:
        if sha in sha_to_n:
            xi = sha_to_n[sha]
            ax2.annotate(
                text,
                (xi, ratio100[xi - 1]),
                xytext=(xi + dx, ty),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
                color="0.25",
            )
    if has_wt:
        wt = redone_by[-1] / max(rows[-1]["added"], 1)
        ax2.scatter(
            [nc + 3],
            [wt],
            marker="*",
            s=180,
            color="#7d3c98",
            zorder=5,
            label=f"uncommitted working tree ({wt:.2f})",
        )
    ax2.set_ylabel("lines redone / new lines\n(rolling)")
    ax2.set_title(
        "Volume: how much recently written code gets thrown away or rewritten", fontsize=11
    )
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    for ax in (ax1, ax2):
        ax.set_xlim(0, nc + 5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("fork commit number (topological order of the current lineage)")
        ax.grid(alpha=0.25)
    ax1.set_ylim(0, np.nanmax(roll(np.array(followups[:nc]) * code, code)) * 1.35)
    top = ax1.secondary_xaxis("top")
    ticks = np.linspace(1, nc, 9).astype(int)
    top.set_xticks(ticks)
    top.set_xticklabels(
        [
            dt.datetime.fromtimestamp(rows[t - 1]["ts"], dt.timezone.utc).strftime("%m-%d")
            for t in ticks
        ],
        fontsize=7.5,
        color="0.45",
    )
    top.set_xlabel(
        "commit date, for orientation only (not proportional to work)", fontsize=8, color="0.45"
    )

    # Panel 3: effort clock. Follow-ups per 1,000 new lines, binned by cumulative new lines.
    lefts, widths, vals = [], [], []
    acc_a = acc_f = 0.0
    start = 0.0
    for i in range(nc):
        acc_a += added[i]
        acc_f += followups[i]
        if acc_a >= args.bin or i == nc - 1:
            lefts.append(start)
            widths.append(cum[i] - start)
            vals.append(1000 * acc_f / max(acc_a, 1))
            start = cum[i]
            acc_a = acc_f = 0.0
    ax3.bar(
        lefts,
        vals,
        width=widths,
        align="edge",
        color="#2c6fbb",
        alpha=0.75,
        edgecolor="white",
        lw=0.5,
    )
    ax3.axvspan(cum[nc - k - 1], cum[-1], color="#2c6fbb", alpha=0.08, lw=0)
    ax3.set_xlim(0, cum[-1] * 1.01)
    ax3.set_ylim(0, max(vals) * 1.3)
    ax3.set_xlabel("effort clock: cumulative new source lines written (thousands)")
    ax3.set_ylabel("follow-up repairs\nper 1,000 new lines")
    ax3.set_title(
        f"Same fix-on-fix signal on the effort clock (bins of ~{args.bin} new lines)", fontsize=11
    )
    ax3.grid(alpha=0.25, axis="y")

    for j, (sha, label) in enumerate(EVENTS):
        if sha in sha_to_n and sha_to_n[sha] <= nc:
            xi = sha_to_n[sha]
            for ax, xv in ((ax1, xi), (ax2, xi), (ax3, cum[xi - 1])):
                ax.axvline(xv, color="0.35", lw=0.9, ls=":")
            for ax, xv in ((ax1, xi), (ax3, cum[xi - 1])):
                ax.text(
                    xv,
                    0.98 if j % 2 == 0 else 0.87,
                    " " + label.replace("\n", "\n "),
                    transform=ax.get_xaxis_transform(),
                    fontsize=7.5,
                    va="top",
                    color="0.25",
                )
    for sha, text in EFFORT_NOTES:
        if sha in sha_to_n and sha_to_n[sha] <= nc:
            xi = sha_to_n[sha]
            ax3.text(
                cum[xi - 1] - added[xi - 1] / 1000 + 0.3,
                0.06,
                text,
                fontsize=7,
                color="0.25",
                transform=ax3.get_xaxis_transform(),
            )
    if third_n and third_n <= nc:
        for ax, (a, b) in (
            (ax1, (third_n, nc)),
            (ax2, (third_n, nc)),
            (ax3, (cum[third_n - 1], cum[-1])),
        ):
            ax.axvspan(a, b, ymin=0, ymax=0.02, color="0.2", lw=0)
        for ax, xv in ((ax1, third_n), (ax3, cum[third_n - 1])):
            ax.text(
                xv,
                0.035,
                "▸ last ⅓ of all repository commits (incl. upstream)",
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                color="0.2",
            )

    path = os.path.join(args.out, "history-churn.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"wrote {path}, history-churn.csv, history-churn-edges.csv")

    # Segment summary between event markers (committed history only).
    marks = [(s, sha_to_n[s] - 1) for s, _ in EVENTS if s in sha_to_n and sha_to_n[s] <= nc]
    cuts = [("start", 0)] + marks + [("end", nc)]
    groups = [(f"from {name}", list(range(a, b))) for (name, a), (_, b) in zip(cuts, cuts[1:])]
    if args.since:
        try:
            in_range = {h[:7] for h in git("rev-list", f"{args.since}..HEAD").split()}
        except subprocess.CalledProcessError:
            raise SystemExit(f"--since: unknown revision {args.since!r}")
        # Exactly the fork commits in REV..HEAD, which need not be contiguous in topo order.
        groups.append((f"{args.since}..HEAD", [i for i in range(nc) if rows[i]["sha"] in in_range]))
    redone_by_a, redone_by100_a = np.array(redone_by[:nc]), np.array(redone_by100[:nc])
    followups_a, corrective_a = np.array(followups[:nc]), np.array(corrective[:nc])
    print(
        f"\n{'segment':<20}{'commits':>8}{'new lines':>11}{'ratio≤' + str(k):>10}{'ratio≤100':>11}"
        f"{'follow-ups/change':>19}{'corrective share':>18}"
    )
    for name, idx in groups:
        if not idx:
            print(f"{name:<20}{0:>8}  (no fork commits)")
            continue
        c = code[idx]
        na = max(added[idx].sum(), 1)
        print(
            f"{name:<20}{len(idx):>8}{int(added[idx].sum()):>11}"
            f"{redone_by_a[idx].sum() / na:>10.2f}{redone_by100_a[idx].sum() / na:>11.2f}"
            f"{(followups_a[idx] * c).sum() / max(c.sum(), 1):>19.2f}"
            f"{(corrective_a[idx] * c).sum() / max(c.sum(), 1):>18.2f}"
        )
    if has_wt:
        print(
            f"uncommitted working tree: +{rows[-1]['added']} lines, "
            f"redoes {redone_by[-1]} recent lines"
        )
    print(
        f"\nnote: follow-ups for the last {k} commits are not yet observable (biased low); "
        "re-run after later work to complete them."
    )


if __name__ == "__main__":
    main()
