import os
import csv
import re
from bs4 import BeautifulSoup

DATA_DIR = 'data'
CSV_FILE = 'data.csv'

def extract_titles_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    for tag in soup.find_all('div', class_="guest-list-circle ttooltip img-circle float-left mar-right15 mar-bot15"):
        title = tag.get('title')
        if title:
            results.append(title)
    return results

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

    # Process each file in data/
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path):
            # Strip .html from filename for header
            header_name = filename
            if header_name.lower().endswith('.html'):
                header_name = header_name[:-5]
            # Skip if already processed
            if header_name in headers:
                continue
            try:
                with open(file_path, encoding='utf-8') as f:
                    html = f.read()
            except UnicodeDecodeError:
                print(f"Warning: Could not read {file_path} as UTF-8. Skipping.")
                continue
            titles = extract_titles_from_html(html)
            # Add new column
            headers.append(header_name)
            # Pad rows if needed
            while len(rows) < len(titles):
                rows.append([''] * (len(headers) - 1))
            # Add data to each row
            for i, title in enumerate(titles):
                if len(rows[i]) < len(headers) - 1:
                    rows[i].extend([''] * (len(headers) - 1 - len(rows[i])))
                rows[i].append(title)
            # Pad remaining rows
            for i in range(len(titles), len(rows)):
                rows[i].append('')

    # Write updated CSV
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

if __name__ == '__main__':
    main()
