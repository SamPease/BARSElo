#!/usr/bin/env python3
"""Extract scored games from HTML files in data/games and append to Sports Elo - Games.csv

Checks for duplicates by (datetime, team1, team2) before inserting.
Formats time as mm/dd/yyyy HH:MM:00 (24-hour) per request.
"""
import csv
import os
import re
from datetime import datetime
from bs4 import BeautifulSoup
from dateutil import parser as dateparser


GAMES_DIR = os.path.join(os.path.dirname(__file__), "games")
CSV_PATH = os.path.join(os.path.dirname(__file__), "Sports Elo - Games.csv")
TEAMS_CSV_PATH = os.path.join(os.path.dirname(__file__), "Sports Elo - Teams.csv")


def find_html_files(directory):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".html") or f.lower().endswith(".htm"):
                yield os.path.join(root, f)


def parse_game_rows_from_html(path):
    """Parse LeagueLobster schedule-match blocks from the saved HTML.

    This looks for <div class="schedule-match" ... data-iso-datetime="..."> and extracts
    .home-team .team-display, .away-team .team-display and .static-scores. Only returns
    games where static-scores contains numeric scores.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        doc = f.read()

    soup = BeautifulSoup(doc, "lxml")
    games = []

    match_divs = soup.find_all("div", class_=lambda c: c and "schedule-match" in c)
    score_re = re.compile(r"(\d+)\s*[:\-–—]\s*(\d+)")

    for mdiv in match_divs:
        iso = mdiv.get("data-iso-datetime") or mdiv.get("data-datetime")
        # prefer iso if present
        if not iso:
            continue
        try:
            dt = dateparser.parse(iso)
        except Exception:
            # try parsing numeric epoch if present
            try:
                epoch = int(mdiv.get("data-datetime"))
                dt = datetime.fromtimestamp(epoch)
            except Exception:
                continue

        # score
        score_span = mdiv.select_one(".static-scores")
        if not score_span:
            # sometimes score appears in visible-print-inline
            score_span = mdiv.select_one(".visible-print-inline")
        if not score_span:
            continue
        score_text = score_span.get_text(" ", strip=True)
        sm = score_re.search(score_text)
        if not sm:
            # skip matches without numeric posted score (e.g., 'V' or empty)
            continue
        s1, s2 = int(sm.group(1)), int(sm.group(2))

        # team names
        home = None
        away = None
        home_el = mdiv.select_one(".home-team .team-display") or mdiv.select_one(".left-team .team-display")
        away_el = mdiv.select_one(".away-team .team-display") or mdiv.select_one(".right-team .team-display")
        if home_el:
            home = clean_team_name(home_el.get_text(" ", strip=True))
        if away_el:
            away = clean_team_name(away_el.get_text(" ", strip=True))

        if not home or not away:
            continue

        games.append({
            "time": dt,
            "team1": home,
            "score1": s1,
            "team2": away,
            "score2": s2,
            "tournament": "",
        })

    return games


def clean_team_name(name):
    # strip common noise
    name = re.sub(r"\s+\|.*$", "", name)  # remove trailing pipes and metadata
    name = re.sub(r"\n|\r", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    # remove score-like fragments
    name = re.sub(r"\d+[-–—]\d+", "", name).strip()
    # remove stray vs/@
    name = re.sub(r"\b(vs?|@)\b\.?", "", name, flags=re.IGNORECASE).strip()
    return name


def read_existing_games(csv_path):
    """Return (existing_keys_set, existing_rows_list, header_fieldnames, time_col_name, inferred_time_format).

    existing_keys contains canonical tuple keys used for duplicate checks. Key is (time_str, team1, team2) where
    time_str is the canonical formatted time if parseable, otherwise the raw time string.
    """
    existing = set()
    rows = []
    header = None
    time_col = None
    inferred_time_format = None

    if not os.path.exists(csv_path):
        return existing, rows, header, time_col, inferred_time_format

    with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        # determine time column name (case-insensitive match)
        for h in header:
            if h and h.strip().lower().startswith('time'):
                time_col = h
                break

        for r in reader:
            rows.append(r)

    # infer time format from first non-empty time value
    sample = None
    if time_col:
        for r in rows:
            t = (r.get(time_col) or '').strip()
            if t:
                sample = t
                break

    if sample:
        # look for pattern like M/D/YYYY H:MM:SS (1-2 digits month/day/hour)
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})$", sample)
        if m:
            month_part, day_part, year_part, hour_part = m.group(1), m.group(2), m.group(3), m.group(4)
            month_fmt = '%m' if month_part.startswith('0') else '%-m'
            day_fmt = '%d' if day_part.startswith('0') else '%-d'
            hour_fmt = '%H' if hour_part.startswith('0') else '%-H'
            inferred_time_format = f"{month_fmt}/{day_fmt}/%Y {hour_fmt}:%M:%S"
        else:
            # fallback: try parsing and use default no-leading-zero format
            inferred_time_format = "%-m/%-d/%Y %-H:%M:%S"

    # Build existing key set using canonical formatted time when possible
    for r in rows:
        raw_time = ''
        if time_col:
            raw_time = (r.get(time_col) or '').strip()
        key_time = raw_time
        if raw_time:
            try:
                parsed = dateparser.parse(raw_time, fuzzy=True)
                if parsed and inferred_time_format:
                    # format to canonical string using inferred_time_format
                    key_time = parsed.strftime(inferred_time_format)
                elif parsed:
                    key_time = parsed.strftime('%-m/%-d/%Y %-H:%M:%S')
            except Exception:
                key_time = raw_time

        # teams may be under different column names
        team1 = (r.get('Team 1') or r.get('team 1') or r.get('team1') or r.get('Team1') or '').strip()
        team2 = (r.get('Team 2') or r.get('team 2') or r.get('team2') or r.get('Team2') or '').strip()
        existing.add((key_time, team1, team2))

    return existing, rows, header, time_col, inferred_time_format


def format_time_for_csv(dt: datetime):
    # Match existing CSV style: M/D/YYYY H:MM:00 (no leading zeros for month/day/hour)
    # Example: '7/12/2025 8:55:00'
    m = dt.month
    d = dt.day
    y = dt.year
    h = dt.hour
    mi = dt.minute
    return f"{m}/{d}/{y} {h}:{mi:02d}:00"


def main():
    html_files = list(find_html_files(GAMES_DIR))
    print(f"Found {len(html_files)} html files to scan in {GAMES_DIR}")

    all_new_games = []
    for p in html_files:
        try:
            g = parse_game_rows_from_html(p)
            if g:
                print(f"{p}: found {len(g)} scored games")
                all_new_games.extend(g)
        except Exception as e:
            print(f"Error parsing {p}: {e}")

    if not all_new_games:
        print("No new scored games found.")
        return
    # Read existing CSV and header info
    existing_keys, existing_rows, existing_header, time_col, inferred_time_format = read_existing_games(CSV_PATH)

    # Filter new games against existing (compare using canonical formatted time string)
    additions = []
    for g in all_new_games:
        if inferred_time_format:
            key_time = g['time'].strftime(inferred_time_format)
        else:
            # fallback to default formatter
            key_time = format_time_for_csv(g['time'])
        key = (key_time, g['team1'].strip(), g['team2'].strip())
        rev_key = (key_time, g['team2'].strip(), g['team1'].strip())
        if key in existing_keys or rev_key in existing_keys:
            print(f"Skipping duplicate game: {g['team1']} vs {g['team2']} at {key_time}")
            continue
        additions.append(g)

    if not additions:
        print("No additions after dedupe check.")
        return

    # Load known team names from Teams CSV header for validation
    def load_known_teams(path):
        if not os.path.exists(path):
            return set()
        try:
            with open(path, newline='', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                first = next(reader, [])
                # clean and include non-empty header cells
                return set(clean_team_name(h).strip() for h in first if (h or '').strip())
        except Exception:
            return set()

    known_teams = load_known_teams(TEAMS_CSV_PATH)
    if not known_teams:
        print(f"Warning: no teams found in {TEAMS_CSV_PATH}; skipping team-existence validation.")

    # Validate additions against known teams; collect skipped games for reporting
    valid_additions = []
    skipped_missing = []
    for g in additions:
        t1 = g['team1'].strip()
        t2 = g['team2'].strip()
        missing = []
        if known_teams and t1 not in known_teams:
            missing.append(t1)
        if known_teams and t2 not in known_teams:
            missing.append(t2)
        if missing:
            # record skipped game with which teams missing
            skipped_missing.append((g, missing))
            print(f"Skipping game because unknown team(s): {t1} vs {t2} at {g['time']} -> missing: {', '.join(missing)}")
            continue
        valid_additions.append(g)

    if not valid_additions:
        print(f"No additions after team-existence validation. {len(skipped_missing)} games skipped due to unknown teams.")
        return

    # Determine header to write: prefer existing header casing if present
    if existing_header:
        header = existing_header
    else:
        header = ['Time', 'Team 1', 'Score 1', 'Team 2', 'Score 2', 'Outcome Only', 'Tournament']

    # Build combined rows: start with existing_rows (preserve raw values), then append additions
    combined_rows = []
    for r in existing_rows:
        combined_rows.append(r.copy())

    # append additions in the CSV column format
    for g in valid_additions:
        if inferred_time_format:
            time_str = g['time'].strftime(inferred_time_format)
        else:
            time_str = format_time_for_csv(g['time'])
        combined_rows.append({
            'Time': time_str,
            'Team 1': g['team1'],
            'Score 1': str(g['score1']),
            'Team 2': g['team2'],
            'Score 2': str(g['score2']),
            'Outcome Only': '',
            'Tournament': g.get('tournament','')
        })

    # Sort combined rows by parsed datetime when possible; rows with unparsable times go to the end in original order
    def sort_key(row):
        t = (row.get(time_col) or row.get('Time') or row.get('time') or '').strip()
        try:
            dt = dateparser.parse(t, fuzzy=True)
            return (0, dt)
        except Exception:
            return (1, t)

    combined_rows.sort(key=sort_key)

    tmp_path = CSV_PATH + ".tmp"
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in combined_rows:
            # Map values to header names where possible, default to empty string
            out = {}
            for h in header:
                out[h] = (r.get(h) or r.get(h.lower()) or r.get(h.replace(' ', '')) or r.get(h.replace(' ', '').lower()) or '').strip()
            writer.writerow(out)

    os.replace(tmp_path, CSV_PATH)
    print(f"Appended {len(additions)} new games and wrote sorted CSV to {CSV_PATH}")


if __name__ == '__main__':
    main()
