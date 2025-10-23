# BARSElo

Data retrival is pretty manual right now. If I got access to more data I would automate it. Then when given games.csv and teams.csv this calculates elo at each timestep and saves to calculated_elo.csv

## Extract scored games from HTML

A helper script `extract_games.py` scans HTML files in `data/games` for games with posted scores and appends them to `Sports Elo - Games.csv`.

Usage:

1. Install dependencies (in a virtualenv):

   pip install -r requirements.txt

2. Run the script from the repository root:

   python extract_games.py

The script will deduplicate by (datetime, team1, team2) and sort the CSV by date (oldest first).
Data retrival is pretty manual right now. If I got access to more data I would automate it. Then when given games.csv and teams.csv this calculates elo at each timestep and saves to calculated_elo.csv
