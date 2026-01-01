# BARSElo

A machine learning-based player rating system for recreational dodgeball that uses Bradley-Terry models with Bayesian uncertainty to rank individual player skill from noisy team game data.

**[View Live Rankings](https://sampease.github.io/rankings.html)** | **[Read Full Project Write-up](https://sampease.github.io/barselo.html)**

## Overview

BARSElo tackles a challenging problem: how do you rank individual player skill when win/loss records are heavily determined by randomly-assigned teams? This project uses historical game data from Big Apple Recreational Sports (BARS) dodgeball leagues in NYC to train statistical models that separate individual skill from team composition effects.

**Key Features:**
- Bradley-Terry batch optimization with Gaussian margin-of-victory likelihood
- Bayesian uncertainty quantification for sparse-data players
- Dual evaluation modes (chronological + new-team generalization)
- Head-to-head win probability rankings (O(n²) vectorized)
- Interactive visualization with player trajectories and team comparisons

**Current Performance:**
- 790+ games, 139 teams, 400+ players
- New-team generalization: 54.4% accuracy, NLL: 0.666
- Cross-mode (chronological): 62.2% accuracy

## Project Structure

```
BARSElo/
├── data/
│   ├── extract_games.py      # Parse HTML for game results
│   ├── extract_teams.py      # Extract team rosters
│   ├── resolve_aliases.py    # Handle player name variations
│   ├── data_loader.py        # Data loading utilities
│   └── Sports Elo - Games.csv  # Game results dataset
├── models/
│   ├── base.py               # Base model class
│   ├── elo.py                # Classic Elo implementation
│   ├── bt_mov.py             # Bradley-Terry with MOV
│   ├── bt_uncert.py          # BT with Bayesian uncertainty (current)
│   └── trueskill.py          # TrueSkill implementation
├── train/
│   ├── unified_train.py      # Training & hyperparameter optimization
│   ├── unified_config.json   # Model configurations
│   └── evaluate_new_team_generalization.py
├── viz/
│   ├── calculate.py          # Compute ratings for visualization
│   └── export_static_data.py # Generate JSON for static site
└── static-site/
    ├── index.html            # Interactive rankings viewer
    └── data/elo_data.json    # Pre-computed ratings
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sampease/BARSElo.git
cd BARSElo
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection

1. Download HTML files from league scheduling site to `data/games/` and `data/teams/`

2. Extract game results:
```bash
python data/extract_games.py
```

3. Extract team rosters:
```bash
python data/extract_teams.py
```

4. Resolve player name aliases:
```bash
python data/resolve_aliases.py
```

### Training Models

Train a specific model with hyperparameter optimization:
```bash
python train/unified_train.py --config train/unified_config.json --mode new_team
```

Available modes:
- `chrono`: Chronological evaluation (online learning simulation)
- `new_team`: New-team generalization (cold-start evaluation)
- `both`: Run both modes with cross-validation

### Generate Rankings

1. Calculate ratings for all models:
```bash
python viz/calculate.py
```

2. Export data for static site:
```bash
python viz/export_static_data.py
```

3. Serve locally:
```bash
cd static-site
python serve.py
```

Then open `http://localhost:8000` in your browser.

## Models

### Current: BT-Uncert (Bradley-Terry with Uncertainty)
- Bayesian uncertainty that decreases with games played (alpha parameter)
- Gaussian margin likelihood (sigma parameter)
- L2 regularization (l2_lambda)
- Rankings via average head-to-head win probability (not μ - k*σ)

### Also Implemented:
- **BT-MOV**: Baseline Bradley-Terry with Gaussian MOV
- **BT-VET**: Veteran-weighted uncertainty (slightly better NLL)
- **BT-MOV-Time-Decay**: Exponential decay on older games
- **Elo**: Classic with MOV multipliers and tournament weighting
- **TrueSkill**: Microsoft's rating system (overfits on this dataset)

## Key Technical Decisions

1. **Batch optimization**: Optimize all player skills simultaneously using entire game history (not sequential updates)
2. **Dual evaluation**: New-team mode prevents models from cheating via reactivity
3. **Gaussian margins**: Empirical distribution of margins is Gaussian
4. **Head-to-head ranking**: Principled compression of (skill, uncertainty) into 1D rankings
5. **Vectorization**: NumPy broadcasting for efficient O(n²) pairwise comparisons

## Data Sources

- League game results: LeagueLobster HTML pages (semi-automated collection)
- Tournament data: Cross-team matchups and travel team performance
- Time span: Fall 2023 - present
- 790+ games, 139 teams, 400+ players
- ~60% of players have ≤15 games (sparse data challenge)

## Links

- **[Live Rankings](https://sampease.github.io/rankings.html)** - Interactive player skill trajectories and team comparisons
- **[Project Write-up](https://sampease.github.io/barselo.html)** - Full technical deep dive and journey
- **[GitHub Repository](https://github.com/sampease/BARSElo)** - This repo

## Tech Stack

- **Python 3.x** - Primary language
- **scipy.optimize** - L-BFGS-B for batch optimization
- **Optuna** - Bayesian hyperparameter optimization
- **NumPy/Pandas** - Numerical computation and data processing
- **BeautifulSoup** - HTML parsing for data extraction
- **Plotly.js** - Interactive visualizations
- **GitHub Pages** - Static site hosting

## License

MIT License - see LICENSE file for details

## Contact

Samantha Pease - [sampease.github.io](https://sampease.github.io)
