#!/usr/bin/env python3
"""Interactive alias resolver for player names in a teams CSV.

Reads a teams CSV (columns = teams, rows = players), extracts all unique player
names, suggests likely misspellings / duplicates using a combination of
last-name grouping and difflib similarity, then asks the user to confirm and
choose a canonical name. Replacements are applied to the teams CSV and
recorded in `aliases.csv` as `alias,canonical`.

Usage:
    python resolve_aliases.py --teams "Sports Elo - Teams.csv" [--dry-run]

The script uses only Python stdlib to avoid extra dependencies.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def normalize(name: str) -> str:
    if not name:
        return ""
    # strip, lowercase, remove diacritics and punctuation except spaces
    s = name.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # replace punctuation with space
    s = re.sub(r"[\'\"\.`\-()]", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def last_name(name: str) -> str:
    n = name.strip()
    if not n:
        return ""
    parts = n.split()
    return parts[-1].lower()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def read_teams_csv(path: str) -> Tuple[List[List[str]], List[str]]:
    """Return (rows, header) where header is first row (team names).
    rows are lists of cell strings (including header)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        all_rows = [row for row in reader]
    if not all_rows:
        return [], []
    header = all_rows[0]
    rows = all_rows[1:]
    return rows, header


def collect_names(rows: List[List[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        for cell in row:
            if cell is None:
                continue
            name = cell.strip()
            if name:
                counts[name] += 1
    return dict(counts)


def build_candidates(names: List[str], counts: Dict[str, int]) -> List[Tuple[str, str, float]]:
    """Return list of candidate pairs (name1, name2, score) sorted desc by score."""
    normalized = {n: normalize(n) for n in names}
    last = {n: last_name(n) for n in names}

    candidates: List[Tuple[str, str, float]] = []
    N = len(names)
    for i in range(N):
        for j in range(i + 1, N):
            a = names[i]
            b = names[j]
            na = normalized[a]
            nb = normalized[b]
            if not na or not nb:
                continue
            score = 0.0
            # exact normalized match (but different raw spelling)
            if na == nb and a != b:
                score = 1.0
            else:
                # last-name equality helps
                if last[a] and last[b] and last[a] == last[b]:
                    # compare first-name (everything except last) similarity
                    fa = " ".join(na.split()[:-1])
                    fb = " ".join(nb.split()[:-1])
                    sim = similarity(fa, fb) if fa or fb else similarity(na, nb)
                    # boost if first-name similar
                    score = 0.7 + 0.3 * sim
                else:
                    # full-name similarity
                    sim = similarity(na, nb)
                    if sim > 0.85:
                        score = sim
                    # substring containment (one is contained in the other)
                    elif na in nb or nb in na:
                        score = max(score, 0.8)
            if score > 0.72:
                candidates.append((a, b, score))

    # sort by score desc then by combined counts desc
    candidates_sorted = sorted(
        candidates,
        key=lambda t: (t[2], counts.get(t[0], 0) + counts.get(t[1], 0)),
        reverse=True,
    )
    return candidates_sorted


def confirm(prompt: str) -> bool:
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def apply_replacements(rows: List[List[str]], replacements: Dict[str, str]) -> List[List[str]]:
    # replace exact cell matches (strip comparison)
    out = []
    for row in rows:
        newrow = []
        for cell in row:
            raw = cell or ""
            key = raw.strip()
            if key in replacements:
                # preserve leading/trailing spaces if any
                leading = raw[: len(raw) - len(raw.lstrip(" "))]
                trailing = raw[len(raw.rstrip(" ")) :]
                newcell = leading + replacements[key] + trailing
            else:
                newcell = raw
            newrow.append(newcell)
        out.append(newrow)
    return out


def append_aliases_csv(path: str, pairs: List[Tuple[str, str]]) -> None:
    exist = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                exist.add((r[0].strip(), r[1].strip()))
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for a, b in pairs:
            entry = (a.strip(), b.strip())
            if entry in exist:
                continue
            writer.writerow(entry)
            exist.add(entry)


def write_teams_csv(path: str, header: List[str], rows: List[List[str]]) -> None:
    # backup first
    bak = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, bak)
    print(f"Backup created at {bak}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve player name aliases in a teams CSV")
    parser.add_argument("--teams", default="Sports Elo - Teams.csv", help="path to teams CSV")
    parser.add_argument("--aliases", default="aliases.csv", help="path to aliases CSV to append")
    parser.add_argument("--dry-run", action="store_true", help="only show suggested merges, don't write")
    parser.add_argument("--min-score", type=float, default=0.72, help="minimum candidate score to consider")
    args = parser.parse_args()

    if not os.path.exists(args.teams):
        print("Teams CSV not found:", args.teams)
        sys.exit(2)

    rows, header = read_teams_csv(args.teams)
    name_counts = collect_names(rows)
    names = sorted(name_counts.keys())
    print(f"Found {len(names)} unique player names (non-empty).")

    candidates = build_candidates(names, name_counts)
    if not candidates:
        print("No candidate duplicates found with current heuristics.")
        return

    print(f"Found {len(candidates)} candidate pairs. Will ask for confirmation interactively.")

    replacements: Dict[str, str] = {}
    alias_pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for a, b, score in candidates:
        if (a, b) in seen or (b, a) in seen:
            continue
        # skip if either already set to be replaced the other way
        if a in replacements and replacements[a] == b:
            continue
        if b in replacements and replacements[b] == a:
            continue

        print("\nCandidate pair (score={:.3f}):".format(score))
        print(f"  1) {a}   ({name_counts.get(a,0)} occurrences)")
        print(f"  2) {b}   ({name_counts.get(b,0)} occurrences)")

        same = confirm("Are these the same person?")
        if not same:
            seen.add((a, b))
            continue

        # ask which to keep
        while True:
            choice = input("Which should be the canonical spelling? (1/2/new/skip): ").strip().lower()
            if choice == "1":
                keep = a
                remove = b
                break
            if choice == "2":
                keep = b
                remove = a
                break
            if choice == "new":
                newname = input("Enter canonical spelling to use: ").strip()
                if newname:
                    keep = newname
                    remove = a
                    # we will map both a and b to newname
                    break
            if choice == "skip":
                keep = None
                break
            print("Please enter 1, 2, new, or skip.")

        if keep is None:
            seen.add((a, b))
            continue

        # record replacements: map the non-kept raw strings to keep
        # if user entered 'new' canonical, map both a and b to it
        if choice == "new":
            replacements[a] = keep
            replacements[b] = keep
            alias_pairs.append((a, keep))
            alias_pairs.append((b, keep))
        else:
            replacements[remove] = keep
            alias_pairs.append((remove, keep))

        # mark seen so we don't re-ask for these two
        seen.add((a, b))

    if not replacements:
        print("No replacements confirmed.")
        return

    print("\nSummary of replacements to apply:")
    for old, new in replacements.items():
        print(f"  '{old}' -> '{new}'")

    if args.dry_run:
        print("Dry-run mode: no files will be modified.")
        return

    # apply replacements to rows
    new_rows = apply_replacements(rows, replacements)
    write_teams_csv(args.teams, header, new_rows)
    append_aliases_csv(args.aliases, alias_pairs)
    print(f"Applied replacements and appended {len(alias_pairs)} alias rows to {args.aliases}.")


if __name__ == "__main__":
    main()
