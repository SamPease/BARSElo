import os
import csv
import re
import argparse
from bs4 import BeautifulSoup

DATA_DIR = os.path.join('data/teams')
CSV_FILE = 'Sports Elo - Teams.csv'

def extract_titles_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    for tag in soup.find_all('div', class_="guest-list-circle ttooltip img-circle float-left mar-right15 mar-bot15"):
        title = tag.get('title')
        if title:
            results.append(title)
    return results

def extract_roster_from_lickitung(html_content):
    """Lickitung's HTML uses a roster list with player names in spans; extract those."""
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    # Each player appears in a roster-row with the player name in a span (see inspection)
    for row in soup.select('.roster-row'):
        # Try to find a span with visible text. CSS classes can contain colons
        # (e.g. 'tw:text-ellipsis'), so avoid using a class selector with a colon.
        name = None
        for sp in row.find_all('span'):
            text = sp.get_text(strip=True)
            if text:
                name = text
                break
        # fallback: find the image title attribute
        if not name:
            img = row.find('img')
            if img and img.get('title'):
                name = img.get('title')
        if name:
            results.append(name)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for n in results:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def clean_header_name(filename):
    """Strip suffixes from filename to get a clean team name.

    Examples handled:
    - 'Articuno ｜ League and Tournament Scheduler by ...'
    - 'TeamName | League ...'
    - 'TeamName (something)'
    - 'TeamName - other'
    Falls back to the filename without extension if no separator found.
    """
    base = os.path.splitext(filename)[0]
    # split on ' | ' or fullwidth ' ｜ ' or hyphen variants with spaces around
    parts = re.split(r'\s(?:[|\uFF5C\-\–\—])\s', base, maxsplit=1)
    name = parts[0]
    # also split off any trailing parenthetical
    name = re.split(r'\s*\(', name, maxsplit=1)[0]
    return name.strip()


def detect_and_extract(html):
    """Detect which extractor to use and return a list of player names."""
    soup = BeautifulSoup(html, 'html.parser')
    # prefer guest-list-circle if present
    if soup.select('.guest-list-circle'):
        return extract_titles_from_html(html)
    # roster rows
    if soup.select('.roster-row'):
        return extract_roster_from_lickitung(html)
    # LeagueLobster registrations page with multiple teams
    # If the page contains a team-list, return the players for the first team
    # (higher-level handler in main will detect and handle multiple teams).
    if soup.select('ul.team-list li.team'):
        # extract players from the first team element to preserve existing API
        team_el = soup.select_one('ul.team-list li.team')
        players = []
        # look for member list items inside the team-members-container (more precise)
        for li in team_el.select('.team-members-container li.list-group-item'):
            name_container = li.select_one('.col-md-4')
            if not name_container:
                continue
            text = name_container.get_text(separator=' ', strip=True)
            if text:
                players.append(text)
        # dedupe preserve order
        seen = set(); deduped = []
        for n in players:
            if n and n not in seen:
                seen.add(n); deduped.append(n)
        return deduped
    # fallback: collect img title attributes used for avatars
    results = []
    for img in soup.find_all('img'):
        t = img.get('title')
        if t:
            results.append(t.strip())
    # dedupe preserve order
    seen = set(); deduped = []
    for n in results:
        if n and n not in seen:
            seen.add(n); deduped.append(n)
    return deduped


def extract_header_from_html(html):
    """Try to extract a clean team name from the page itself.

    Preference order:
    - the first <h1> text (common in these saved pages)
    - the <title> tag (strip trailing site suffix)
    - fallback to None
    """
    soup = BeautifulSoup(html, 'html.parser')
    # try h1
    h1 = soup.find('h1')
    if h1:
        text = h1.get_text(separator=' ', strip=True)
        # remove any small/child text like <small>
        # text often like 'Purple P***y Eaters <small>' so above covers
        if text:
            return text
    # try title tag and strip common suffix
    title_tag = soup.find('title')
    if title_tag and title_tag.string:
        t = title_tag.string
        # strip trailing ' | League and Tournament Scheduler by ...'
        t = re.split(r'\s\|\sLeague and Tournament Scheduler', t, maxsplit=1)[0]
        return t.strip()
    return None


def normalize_for_lookup(name):
    """Normalize a name for alias lookup: lower, strip punctuation and excess whitespace."""
    if not name:
        return ''
    # lower case
    n = name.lower()
    # replace punctuation with space
    n = re.sub(r"[\W_]+", ' ', n, flags=re.UNICODE)
    # collapse whitespace
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def is_placeholder_name(name):
    """Return True if the extracted text is a UI placeholder or subheader, not a real player name."""
    if not name:
        return True
    n = normalize_for_lookup(name)
    # exact unwanted values
    unwanted = {
        normalize_for_lookup('add'),
        normalize_for_lookup('parent reserve player captain coach'),
    }
    if n in unwanted:
        return True
    # phrases that may contain variable parts
    if 'create new user' in n or 'search player' in n:
        return True
    return False


def load_aliases(alias_file_path):
    """Load aliases CSV of the form: alias,preferred

    Returns a dict mapping normalized alias -> preferred name.
    If file doesn't exist, returns empty dict.
    """
    mapping = {}
    if not alias_file_path:
        return mapping
    if not os.path.exists(alias_file_path):
        return mapping
    try:
        with open(alias_file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # allow files with header or with 2+ columns
                alias = row[0].strip()
                preferred = row[1].strip() if len(row) > 1 else alias
                if alias:
                    mapping[normalize_for_lookup(alias)] = preferred
    except Exception:
        # on any read error, return empty mapping
        return {}
    return mapping


def apply_aliases_and_dedupe(names, alias_map):
    """Map extracted names via alias_map and dedupe while preserving order.

    alias_map: normalized alias -> preferred name
    """
    mapped = []
    for n in names:
        key = normalize_for_lookup(n)
        if key in alias_map:
            mapped_name = alias_map[key]
        else:
            mapped_name = n
        mapped.append(mapped_name)

    # dedupe preserving order
    seen = set()
    deduped = []
    for n in mapped:
        if n and n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped

def main():
    # Read existing CSV
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='', encoding='utf-8') as f:
            reader = list(csv.reader(f))
        headers = reader[0] if reader else []
        rows = reader[1:] if len(reader) > 1 else []
    else:
        headers = []
        rows = []

    # Determine folder to process from CLI args
    parser = argparse.ArgumentParser(description='Extract team rosters from HTML files and append to CSV')
    parser.add_argument('--folder', '-f', default=DATA_DIR, help='Folder containing HTML team files')
    parser.add_argument('--aliases', '-a', default='aliases.csv', help="CSV file with alias,preferred per line. If missing, no aliases applied")
    args = parser.parse_args()
    target_dir = args.folder
    alias_file = args.aliases
    alias_map = load_aliases(alias_file)

    # Process all .html files in the target folder
    for filename in sorted(os.listdir(target_dir)):
        if not filename.lower().endswith('.html'):
            continue
        file_path = os.path.join(target_dir, filename)
        if not os.path.isfile(file_path):
            continue

        # Read HTML early so we can extract the header from the page itself
        try:
            with open(file_path, encoding='utf-8') as f:
                html = f.read()
        except UnicodeDecodeError:
            print(f"Warning: Could not read {file_path} as UTF-8. Skipping.")
            continue

        # Parse the HTML to detect whether this file contains multiple teams
        soup = BeautifulSoup(html, 'html.parser')
        teams_on_page = soup.select('ul.team-list li.team')

        if teams_on_page:
            # The page lists multiple teams (registrations page). Handle each team as its own column.
            for team_el in teams_on_page:
                # If the team element contains a subheader that indicates an Open Draft
                # skip the entire team (user requested skipping 'Open Draft Spring25').
                team_texts = [t.get_text(separator=' ', strip=True).lower() for t in team_el.find_all() if t.get_text(strip=True)]
                if 'open draft spring25'.lower() in team_texts:
                    # skip this team entirely
                    continue
                # header name: prefer .reg-team-name, fall back to heading text
                team_name_el = team_el.select_one('.reg-team-name') or team_el.select_one('.list-group-item-heading')
                if team_name_el:
                    header_name = team_name_el.get_text(separator=' ', strip=True)
                else:
                    # fallback to filename-derived prefix plus id
                    header_name = clean_header_name(filename)

                # extract players within this team element (precise selector)
                players = []
                for li in team_el.select('.team-members-container li.list-group-item'):
                    name_container = li.select_one('.col-md-4')
                    if not name_container:
                        continue
                    text = name_container.get_text(separator=' ', strip=True)
                    if not text:
                        continue
                    # skip placeholder strings like 'Add' or 'Search player or create a new one'
                    if is_placeholder_name(text):
                        continue
                    players.append(text)

                # filter out obvious non-player lines and apply aliases/dedupe
                # skip items equal to the team header, 'Add', and common subheaders
                forbidden_subheaders = {"open draft spring25", "wtnb", "draft", "social"}
                filtered = []
                for p in players:
                    pn = p.strip()
                    if not pn:
                        continue
                    # skip 'Add' placeholder
                    if pn.lower() == 'add':
                        continue
                    # skip if matches header name
                    if header_name and pn == header_name:
                        continue
                    # skip if looks like a subheader
                    if pn.lower() in forbidden_subheaders:
                        continue
                    filtered.append(pn)

                if alias_map:
                    players = apply_aliases_and_dedupe(filtered, alias_map)
                else:
                    # dedupe preserving order
                    seen = set(); deduped = []
                    for n in filtered:
                        if n and n not in seen:
                            seen.add(n); deduped.append(n)
                    players = deduped

                # write this team column like before
                if header_name in headers:
                    col_idx = headers.index(header_name)
                    print(f"Overwriting column '{header_name}' (index {col_idx}) in {CSV_FILE} -- players={len(players)} {players[:3]}...{players[-3:]}")
                    for r in rows:
                        while len(r) <= col_idx:
                            r.append('')
                    for i, title in enumerate(players):
                        if i >= len(rows):
                            while len(rows) <= i:
                                rows.append([''] * len(headers))
                        rows[i][col_idx] = title
                    # Clear any cells below the new titles for this column
                    for i in range(len(players), len(rows)):
                        rows[i][col_idx] = ''
                else:
                    print(f"Appending new column '{header_name}' to {CSV_FILE} -- players={len(players)} {players[:3]}...{players[-3:]}")
                    headers.append(header_name)
                    while len(rows) < len(players):
                        rows.append([''] * (len(headers) - 1))
                    for i, title in enumerate(players):
                        if len(rows[i]) < len(headers) - 1:
                            rows[i].extend([''] * (len(headers) - 1 - len(rows[i])))
                        rows[i].append(title)
                    for i in range(len(players), len(rows)):
                        rows[i].append('')
        else:
            # legacy single-team-per-file behavior
            header_from_page = extract_header_from_html(html)
            if header_from_page:
                header_name = header_from_page
            else:
                header_name = clean_header_name(filename)
            titles = detect_and_extract(html)
            # apply aliases mapping and dedupe
            if alias_map:
                titles = apply_aliases_and_dedupe(titles, alias_map)
            else:
                # ensure deduped even when no aliases are present
                seen = set(); deduped = []
                for n in titles:
                    if n and n not in seen:
                        seen.add(n); deduped.append(n)
                titles = deduped

            # If header already exists, overwrite that column; otherwise append
            if header_name in headers:
                col_idx = headers.index(header_name)
                print(f"Overwriting column '{header_name}' (index {col_idx}) in {CSV_FILE}")
                # Ensure rows have enough columns
                for r in rows:
                    while len(r) <= col_idx:
                        r.append('')
                # Write titles into the column, extending rows if necessary
                for i, title in enumerate(titles):
                    if i >= len(rows):
                        # create new rows up to i
                        while len(rows) <= i:
                            rows.append([''] * len(headers))
                    rows[i][col_idx] = title
                # Clear cells below titles in that column for rows beyond titles
                for i in range(len(titles), len(rows)):
                    rows[i][col_idx] = ''
            else:
                # Append new column header
                print(f"Appending new column '{header_name}' to {CSV_FILE}")
                headers.append(header_name)
                # Ensure rows list is long enough to hold all player entries
                while len(rows) < len(titles):
                    rows.append([''] * (len(headers) - 1))
                # Add data to each row
                for i, title in enumerate(titles):
                    if len(rows[i]) < len(headers) - 1:
                        rows[i].extend([''] * (len(headers) - 1 - len(rows[i])))
                    rows[i].append(title)
                # For remaining existing rows, append empty cell for this new column
                for i in range(len(titles), len(rows)):
                    rows[i].append('')

    # Write updated CSV
    # Clean any lingering placeholder UI strings from the table before writing.
    for r_idx, r in enumerate(rows):
        for c_idx, cell in enumerate(r):
            if is_placeholder_name(cell):
                rows[r_idx][c_idx] = ''

    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

if __name__ == '__main__':
    main()
