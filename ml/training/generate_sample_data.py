"""Generate a bootstrap Premier League valuation dataset for MVP development.

Replace data/raw/premier_league_valuations.csv with real labelled valuations
before treating predictions as serious market estimates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "raw" / "premier_league_valuations.csv"

POSITIONS = ["GK", "DEF", "MID", "FWD"]
CLUBS = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle",
    "Nottingham Forest",
    "Tottenham",
    "West Ham",
    "Wolves",
    "Leicester",
    "Leeds",
    "Southampton",
]

# Bootstrap PL-style roster for searchable demo names.
# Stats/values are synthesized — swap for real labelled data later (see DATA.md).
SEED_PLAYERS = [
    # Arsenal
    ("David Raya", "GK", "Arsenal", 28),
    ("William Saliba", "DEF", "Arsenal", 23),
    ("Gabriel Magalhaes", "DEF", "Arsenal", 26),
    ("Jurrien Timber", "DEF", "Arsenal", 23),
    ("Riccardo Calafiori", "DEF", "Arsenal", 22),
    ("Ben White", "DEF", "Arsenal", 26),
    ("Declan Rice", "MID", "Arsenal", 25),
    ("Martin Odegaard", "MID", "Arsenal", 25),
    ("Thomas Partey", "MID", "Arsenal", 31),
    ("Kai Havertz", "MID", "Arsenal", 25),
    ("Bukayo Saka", "MID", "Arsenal", 22),
    ("Gabriel Martinelli", "FWD", "Arsenal", 23),
    ("Leandro Trossard", "FWD", "Arsenal", 29),
    ("Gabriel Jesus", "FWD", "Arsenal", 27),
    # Aston Villa
    ("Emiliano Martinez", "GK", "Aston Villa", 32),
    ("Ezri Konsa", "DEF", "Aston Villa", 26),
    ("Pau Torres", "DEF", "Aston Villa", 27),
    ("Lucas Digne", "DEF", "Aston Villa", 31),
    ("Youri Tielemans", "MID", "Aston Villa", 27),
    ("John McGinn", "MID", "Aston Villa", 29),
    ("Amadou Onana", "MID", "Aston Villa", 23),
    ("Morgan Rogers", "MID", "Aston Villa", 22),
    ("Ollie Watkins", "FWD", "Aston Villa", 28),
    ("Jhon Duran", "FWD", "Aston Villa", 21),
    # Bournemouth
    ("Neto", "GK", "Bournemouth", 35),
    ("Illia Zabarnyi", "DEF", "Bournemouth", 22),
    ("Marcos Senesi", "DEF", "Bournemouth", 27),
    ("Lewis Cook", "MID", "Bournemouth", 27),
    ("Ryan Christie", "MID", "Bournemouth", 29),
    ("Antoine Semenyo", "FWD", "Bournemouth", 24),
    ("Justin Kluivert", "FWD", "Bournemouth", 25),
    ("Evanilson", "FWD", "Bournemouth", 24),
    # Brentford
    ("Mark Flekken", "GK", "Brentford", 31),
    ("Nathan Collins", "DEF", "Brentford", 23),
    ("Ethan Pinnock", "DEF", "Brentford", 31),
    ("Christian Norgaard", "MID", "Brentford", 30),
    ("Vitaly Janelt", "MID", "Brentford", 26),
    ("Bryan Mbeumo", "FWD", "Brentford", 25),
    ("Yoane Wissa", "FWD", "Brentford", 28),
    ("Kevin Schade", "FWD", "Brentford", 22),
    # Brighton
    ("Bart Verbruggen", "GK", "Brighton", 22),
    ("Lewis Dunk", "DEF", "Brighton", 32),
    ("Pervis Estupinan", "DEF", "Brighton", 26),
    ("Joao Pedro", "FWD", "Brighton", 23),
    ("Kaoru Mitoma", "MID", "Brighton", 27),
    ("Danny Welbeck", "FWD", "Brighton", 33),
    ("Pascal Gross", "MID", "Brighton", 33),
    ("Yankuba Minteh", "FWD", "Brighton", 20),
    # Chelsea
    ("Robert Sanchez", "GK", "Chelsea", 27),
    ("Levi Colwill", "DEF", "Chelsea", 21),
    ("Reece James", "DEF", "Chelsea", 24),
    ("Malo Gusto", "DEF", "Chelsea", 21),
    ("Wesley Fofana", "DEF", "Chelsea", 23),
    ("Moises Caicedo", "MID", "Chelsea", 22),
    ("Enzo Fernandez", "MID", "Chelsea", 23),
    ("Cole Palmer", "MID", "Chelsea", 22),
    ("Noni Madueke", "FWD", "Chelsea", 22),
    ("Nicolas Jackson", "FWD", "Chelsea", 23),
    ("Christopher Nkunku", "FWD", "Chelsea", 26),
    ("Pedro Neto", "FWD", "Chelsea", 24),
    # Crystal Palace
    ("Dean Henderson", "GK", "Crystal Palace", 27),
    ("Marc Guehi", "DEF", "Crystal Palace", 24),
    ("Daniel Munoz", "DEF", "Crystal Palace", 28),
    ("Adam Wharton", "MID", "Crystal Palace", 20),
    ("Eberechi Eze", "MID", "Crystal Palace", 26),
    ("Jean-Philippe Mateta", "FWD", "Crystal Palace", 27),
    ("Michael Olise", "MID", "Crystal Palace", 22),
    # Everton
    ("Jordan Pickford", "GK", "Everton", 30),
    ("Jarrad Branthwaite", "DEF", "Everton", 22),
    ("James Tarkowski", "DEF", "Everton", 31),
    ("Vitalii Mykolenko", "DEF", "Everton", 25),
    ("Idrissa Gueye", "MID", "Everton", 35),
    ("Abdoulaye Doucoure", "MID", "Everton", 31),
    ("Dwight McNeil", "MID", "Everton", 24),
    ("Dominic Calvert-Lewin", "FWD", "Everton", 27),
    ("Beto", "FWD", "Everton", 26),
    # Fulham
    ("Bernd Leno", "GK", "Fulham", 32),
    ("Antonee Robinson", "DEF", "Fulham", 27),
    ("Issa Diop", "DEF", "Fulham", 27),
    ("Joao Palhinha", "MID", "Fulham", 29),
    ("Andreas Pereira", "MID", "Fulham", 28),
    ("Alex Iwobi", "MID", "Fulham", 28),
    ("Raul Jimenez", "FWD", "Fulham", 33),
    ("Rodrigo Muniz", "FWD", "Fulham", 23),
    # Liverpool
    ("Alisson", "GK", "Liverpool", 32),
    ("Virgil van Dijk", "DEF", "Liverpool", 33),
    ("Ibrahima Konate", "DEF", "Liverpool", 25),
    ("Trent Alexander-Arnold", "DEF", "Liverpool", 26),
    ("Andrew Robertson", "DEF", "Liverpool", 30),
    ("Alexis Mac Allister", "MID", "Liverpool", 25),
    ("Dominik Szoboszlai", "MID", "Liverpool", 23),
    ("Ryan Gravenberch", "MID", "Liverpool", 22),
    ("Curtis Jones", "MID", "Liverpool", 23),
    ("Mohamed Salah", "FWD", "Liverpool", 32),
    ("Luis Diaz", "FWD", "Liverpool", 27),
    ("Darwin Nunez", "FWD", "Liverpool", 25),
    ("Cody Gakpo", "FWD", "Liverpool", 25),
    ("Diogo Jota", "FWD", "Liverpool", 27),
    # Manchester City
    ("Ederson", "GK", "Manchester City", 31),
    ("Ruben Dias", "DEF", "Manchester City", 27),
    ("John Stones", "DEF", "Manchester City", 30),
    ("Josko Gvardiol", "DEF", "Manchester City", 22),
    ("Kyle Walker", "DEF", "Manchester City", 34),
    ("Rodri", "MID", "Manchester City", 28),
    ("Kevin De Bruyne", "MID", "Manchester City", 33),
    ("Bernardo Silva", "MID", "Manchester City", 30),
    ("Phil Foden", "MID", "Manchester City", 24),
    ("Erling Haaland", "FWD", "Manchester City", 24),
    ("Jeremy Doku", "FWD", "Manchester City", 22),
    ("Jack Grealish", "MID", "Manchester City", 29),
    ("Savinho", "FWD", "Manchester City", 20),
    # Manchester United
    ("Andre Onana", "GK", "Manchester United", 28),
    ("Lisandro Martinez", "DEF", "Manchester United", 26),
    ("Harry Maguire", "DEF", "Manchester United", 31),
    ("Luke Shaw", "DEF", "Manchester United", 29),
    ("Diogo Dalot", "DEF", "Manchester United", 25),
    ("Kobbie Mainoo", "MID", "Manchester United", 19),
    ("Bruno Fernandes", "MID", "Manchester United", 30),
    ("Casemiro", "MID", "Manchester United", 32),
    ("Manuel Ugarte", "MID", "Manchester United", 23),
    ("Marcus Rashford", "FWD", "Manchester United", 26),
    ("Alejandro Garnacho", "FWD", "Manchester United", 20),
    ("Rasmus Hojlund", "FWD", "Manchester United", 21),
    ("Amad Diallo", "FWD", "Manchester United", 22),
    # Newcastle
    ("Nick Pope", "GK", "Newcastle", 32),
    ("Sven Botman", "DEF", "Newcastle", 24),
    ("Fabian Schar", "DEF", "Newcastle", 32),
    ("Dan Burn", "DEF", "Newcastle", 32),
    ("Kieran Trippier", "DEF", "Newcastle", 34),
    ("Bruno Guimaraes", "MID", "Newcastle", 26),
    ("Joelinton", "MID", "Newcastle", 28),
    ("Anthony Gordon", "FWD", "Newcastle", 23),
    ("Alexander Isak", "FWD", "Newcastle", 25),
    ("Harvey Barnes", "FWD", "Newcastle", 26),
    ("Jacob Murphy", "FWD", "Newcastle", 29),
    # Nottingham Forest
    ("Matz Sels", "GK", "Nottingham Forest", 32),
    ("Murillo", "DEF", "Nottingham Forest", 22),
    ("Nikola Milenkovic", "DEF", "Nottingham Forest", 26),
    ("Neco Williams", "DEF", "Nottingham Forest", 23),
    ("Morgan Gibbs-White", "MID", "Nottingham Forest", 24),
    ("Elliot Anderson", "MID", "Nottingham Forest", 22),
    ("Callum Hudson-Odoi", "FWD", "Nottingham Forest", 24),
    ("Chris Wood", "FWD", "Nottingham Forest", 32),
    ("Anthony Elanga", "FWD", "Nottingham Forest", 22),
    # Tottenham
    ("Guglielmo Vicario", "GK", "Tottenham", 27),
    ("Cristian Romero", "DEF", "Tottenham", 26),
    ("Micky van de Ven", "DEF", "Tottenham", 23),
    ("Destiny Udogie", "DEF", "Tottenham", 21),
    ("Pedro Porro", "DEF", "Tottenham", 25),
    ("Yves Bissouma", "MID", "Tottenham", 28),
    ("Pape Matar Sarr", "MID", "Tottenham", 22),
    ("James Maddison", "MID", "Tottenham", 27),
    ("Dejan Kulusevski", "MID", "Tottenham", 24),
    ("Son Heung-min", "FWD", "Tottenham", 32),
    ("Brennan Johnson", "FWD", "Tottenham", 23),
    ("Dominic Solanke", "FWD", "Tottenham", 27),
    ("Richarlison", "FWD", "Tottenham", 27),
    # West Ham
    ("Alphonse Areola", "GK", "West Ham", 31),
    ("Kurt Zouma", "DEF", "West Ham", 29),
    ("Aaron Wan-Bissaka", "DEF", "West Ham", 26),
    ("Lucas Paqueta", "MID", "West Ham", 27),
    ("Tomas Soucek", "MID", "West Ham", 29),
    ("Jarrod Bowen", "FWD", "West Ham", 27),
    ("Mohammed Kudus", "MID", "West Ham", 24),
    ("Michail Antonio", "FWD", "West Ham", 34),
    # Wolves
    ("Jose Sa", "GK", "Wolves", 31),
    ("Max Kilman", "DEF", "Wolves", 27),
    ("Rayan Ait-Nouri", "DEF", "Wolves", 23),
    ("Joao Gomes", "MID", "Wolves", 23),
    ("Matheus Cunha", "FWD", "Wolves", 25),
    ("Hwang Hee-chan", "FWD", "Wolves", 28),
    ("Jorgen Strand Larsen", "FWD", "Wolves", 24),
    # Also include a few recent/former PL names users may try
    ("Ivan Toney", "FWD", "Brentford", 28),
    ("Harry Kane", "FWD", "Tottenham", 31),
    ("Raheem Sterling", "FWD", "Chelsea", 29),
    ("Mason Mount", "MID", "Manchester United", 25),
    ("Jadon Sancho", "FWD", "Manchester United", 24),
    ("Kalvin Phillips", "MID", "Manchester City", 28),
    ("Conor Gallagher", "MID", "Chelsea", 24),
    ("James Ward-Prowse", "MID", "West Ham", 29),
    ("Wilfried Zaha", "FWD", "Crystal Palace", 31),
    ("Jamie Vardy", "FWD", "Leicester", 37),
    ("Wilfred Ndidi", "MID", "Leicester", 27),
    ("Kiernan Dewsbury-Hall", "MID", "Leicester", 25),
]


def _position_multiplier(position: str) -> float:
    return {"GK": 0.75, "DEF": 0.9, "MID": 1.05, "FWD": 1.2}[position]


def _age_curve(age: int) -> float:
    # Peak around 25–27
    return float(np.exp(-0.5 * ((age - 26) / 6.5) ** 2))


def _club_prestige(club: str) -> float:
    big_six = {
        "Arsenal",
        "Chelsea",
        "Liverpool",
        "Manchester City",
        "Manchester United",
        "Tottenham",
    }
    if club in big_six:
        return 1.25
    if club in {"Newcastle", "Aston Villa", "Brighton"}:
        return 1.1
    return 1.0


def synthesize_row(
    rng: np.random.Generator,
    player_id: int,
    name: str,
    position: str,
    club: str,
    age: int,
    season: str,
) -> dict:
    if position == "GK":
        appearances = int(rng.integers(18, 38))
        minutes = appearances * int(rng.integers(85, 95))
        goals = int(rng.integers(0, 2))
        assists = int(rng.integers(0, 3))
    elif position == "DEF":
        appearances = int(rng.integers(15, 38))
        minutes = appearances * int(rng.integers(70, 95))
        goals = int(rng.integers(0, 6))
        assists = int(rng.integers(0, 8))
    elif position == "MID":
        appearances = int(rng.integers(18, 38))
        minutes = appearances * int(rng.integers(65, 95))
        goals = int(rng.integers(1, 16))
        assists = int(rng.integers(2, 18))
    else:
        appearances = int(rng.integers(18, 38))
        minutes = appearances * int(rng.integers(60, 95))
        goals = int(rng.integers(4, 30))
        assists = int(rng.integers(1, 14))

    minutes = max(minutes, 1)
    goals_per90 = goals / minutes * 90
    assists_per90 = assists / minutes * 90

    # Keep most values in a realistic PL range; stars can still climb high.
    base = 2_200_000
    value = (
        base
        * _age_curve(age)
        * _position_multiplier(position)
        * _club_prestige(club)
        * (1 + 0.045 * appearances)
        * (1 + 0.12 * goals)
        * (1 + 0.08 * assists)
        * (1 + minutes / 7000)
    )
    value *= float(rng.lognormal(mean=0.0, sigma=0.18))
    value = float(np.clip(value, 400_000, 150_000_000))

    year = int(season.split("/")[0])
    return {
        "player_id": player_id,
        "player_name": name,
        "season": season,
        "age": age,
        "position": position,
        "club": club,
        "appearances": appearances,
        "goals": goals,
        "assists": assists,
        "minutes": minutes,
        "goals_per90": round(goals_per90, 3),
        "assists_per90": round(assists_per90, 3),
        "market_value_eur": int(round(value, -4)),  # nearest 10k
        "valuation_date": f"{year + 1}-06-01",
    }


def generate(n_extra: int = 0, seed: int = 42) -> pd.DataFrame:
    """Build the bootstrap dataset. Prefer named SEED_PLAYERS; n_extra adds fillers."""
    rng = np.random.default_rng(seed)
    seasons = ["2021/22", "2022/23", "2023/24", "2024/25"]
    rows: list[dict] = []
    player_id = 1
    seen_names: set[str] = set()

    for name, position, club, base_age in SEED_PLAYERS:
        if name in seen_names:
            continue
        seen_names.add(name)
        for i, season in enumerate(seasons):
            age = max(16, base_age - (len(seasons) - 1 - i))
            rows.append(
                synthesize_row(rng, player_id, name, position, club, age, season)
            )
        player_id += 1

    if n_extra > 0:
        first_names = [
            "James", "Luke", "Callum", "Harvey", "Mason", "Reece", "Tyler",
            "Adam", "Jacob", "Ethan", "Kai", "Morgan", "Ryan", "Owen", "Nathan",
        ]
        last_names = [
            "Brooks", "Carter", "Dixon", "Ellis", "Foster", "Hayes", "Keller",
            "Lawson", "Mills", "Norris", "Patel", "Quinn", "Reed", "Singh", "Walsh",
        ]
        for _ in range(n_extra):
            name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
            if name in seen_names:
                continue
            seen_names.add(name)
            position = str(rng.choice(POSITIONS, p=[0.12, 0.32, 0.34, 0.22]))
            club = str(rng.choice(CLUBS))
            base_age = int(rng.integers(18, 34))
            n_seasons = int(rng.integers(1, 4))
            chosen = sorted(rng.choice(seasons, size=n_seasons, replace=False))
            for i, season in enumerate(chosen):
                age = base_age + i
                rows.append(
                    synthesize_row(rng, player_id, name, position, club, age, season)
                )
            player_id += 1

    df = pd.DataFrame(rows)
    df = df.sort_values(["player_name", "season"]).reset_index(drop=True)
    return df


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_PATH}")
    print(df[["player_name", "season", "position", "market_value_eur"]].head(8))


if __name__ == "__main__":
    main()
