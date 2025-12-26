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


def build_candidates(names: List[str], counts: Dict[str, int], min_score: float = 0.72) -> List[Tuple[str, str, float]]:
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
            if score > min_score:
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


def write_aliases_csv_full(path: str, all_names: List[str], components: List[Tuple[List[str], str]]) -> None:
    """Overwrite aliases CSV with rows: <aliases_joined> , <canonical>

    - aliases_joined is a pipe-separated list of all variant names for the group.
    - canonical is the chosen canonical spelling for the group.

    We include singleton names as groups of size 1 (aliases==[name], canonical==name)
    so every name appears in the output as the user requested.
    """
    # backup existing file if exists
    if os.path.exists(path):
        bak = f"{path}.bak.{int(time.time())}"
        shutil.copy2(path, bak)
        print(f"aliases CSV backup created at {bak}")

    # Build a map from canonical -> sorted alias list for stable output
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # write a header for clarity (optional)
        writer.writerow(["aliases_pipe_separated", "canonical"])
        # write the provided components first
        for aliases, canonical in components:
            # ensure stable ordering
            uniq = sorted(dict.fromkeys(aliases))
            writer.writerow(["|".join(uniq), canonical])
        # include any leftover singletons not present in components
        present = set()
        for aliases, canonical in components:
            for a in aliases:
                present.add(a)
        for n in sorted(all_names):
            if n not in present:
                writer.writerow([n, n])


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

    candidates = build_candidates(names, name_counts, args.min_score)
    if not candidates:
        print("No candidate duplicates found with current heuristics.")
        # still produce an aliases CSV that maps each name to itself
        if not args.dry_run:
            write_aliases_csv_full(args.aliases, names, [])
            print(f"Wrote aliases CSV at {args.aliases} (no groups found)")
        return

    # Build adjacency graph from candidate pairs (transitive edges)
    adj: Dict[str, Set[str]] = defaultdict(set)
    for a, b, score in candidates:
        adj[a].add(b)
        adj[b].add(a)

    # find connected components
    seen_nodes: Set[str] = set()
    components: List[List[str]] = []
    for n in names:
        if n in seen_nodes:
            continue
        # BFS/DFS to collect component
        stack = [n]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in seen_nodes:
                continue
            seen_nodes.add(cur)
            comp.append(cur)
            for nb in adj.get(cur, []):
                if nb not in seen_nodes:
                    stack.append(nb)
        components.append(comp)

    # prepare final components with canonical selection and interactive confirmation
    final_components: List[Tuple[List[str], str]] = []
    replacements: Dict[str, str] = {}
    for comp in components:
        if len(comp) == 1:
            # singleton; no aliasing needed but still include in output
            final_components.append((comp, comp[0]))
            continue

        # sort comp for stable presentation
        comp_sorted = sorted(comp, key=lambda x: (-name_counts.get(x, 0), x))
        # default canonical = most frequent name
        default_canon = comp_sorted[0]

        print("\nAlias group candidates:")
        for i, nm in enumerate(comp_sorted, start=1):
            print(f"  {i}) {nm}   ({name_counts.get(nm,0)} occurrences)")
        print(f"Default canonical candidate: '{default_canon}'")

        pick = input("Press Enter to accept default, choose a number, enter 'new' to type a canonical name, or 'skip' to leave unchanged: ").strip()
        if pick == "":
            canonical = default_canon
        elif pick.lower() == "skip":
            # leave them as-is (each maps to itself)
            for nm in comp_sorted:
                final_components.append(([nm], nm))
            continue
        elif pick.lower() == "new":
            newname = input("Enter canonical spelling to use: ").strip()
            canonical = newname if newname else default_canon
        else:
            try:
                idx = int(pick)
                if 1 <= idx <= len(comp_sorted):
                    canonical = comp_sorted[idx - 1]
                else:
                    print("Invalid selection; using default.")
                    canonical = default_canon
            except Exception:
                print("Invalid input; using default.")
                canonical = default_canon

        # record component mapping
        final_components.append((comp_sorted, canonical))
        for nm in comp_sorted:
            replacements[nm] = canonical

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
    # prepare alias pairs to append: only write pairs where alias != canonical
    alias_pairs: List[Tuple[str, str]] = []
    for aliases, canonical in final_components:
        for a in aliases:
            if a != canonical:
                alias_pairs.append((a, canonical))

    if alias_pairs:
        append_aliases_csv(args.aliases, alias_pairs)
        print(f"Applied replacements and appended {len(alias_pairs)} alias rows to {args.aliases}.")
    else:
        print("Applied replacements; no non-identity alias rows to append to aliases CSV.")


if __name__ == "__main__":
    main()
