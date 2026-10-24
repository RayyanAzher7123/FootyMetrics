from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.schemas.matches import HeadToHeadSummary, MatchPreviewResponse, TeamForm

FORM_WINDOW = 5


@lru_cache(maxsize=1)
def _load_matches() -> pd.DataFrame:
    path = settings.matches_csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"Match dataset not found at {path}. "
            "Run: python -m ml.training.ingest_football_data"
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def reload_store() -> None:
    _load_matches.cache_clear()


def list_teams() -> list[str]:
    df = _load_matches()
    # Prefer current/recent season teams for the picker
    latest_season = df["season"].iloc[-1]
    recent = df[df["season"] == latest_season]
    teams = sorted(set(recent["home_team"]) | set(recent["away_team"]))
    if len(teams) < 10:
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return teams


def _points(result: str, home: bool) -> int:
    if result == "D":
        return 1
    if home:
        return 3 if result == "H" else 0
    return 3 if result == "A" else 0


def _team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    return df[(df["home_team"] == team) | (df["away_team"] == team)].copy()


def _avg(series: pd.Series, default: float) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return default
    return float(series.mean())


def team_form(team: str) -> TeamForm:
    df = _load_matches()
    season = str(df["season"].max())
    # Current campaign only, so August form is not last season's run-in.
    current = df[df["season"] == season]
    games = _team_matches(current, team).tail(FORM_WINDOW)
    if games.empty:
        return TeamForm(
            team=team,
            form_ppg=1.0,
            goals_for_avg=1.2,
            goals_against_avg=1.2,
            last_results=[],
            form_season=season,
        )

    points: list[int] = []
    gf: list[int] = []
    ga: list[int] = []
    results: list[str] = []
    for row in games.itertuples(index=False):
        is_home = row.home_team == team
        pts = _points(row.result, is_home)
        points.append(pts)
        gf.append(row.home_goals if is_home else row.away_goals)
        ga.append(row.away_goals if is_home else row.home_goals)
        results.append("W" if pts == 3 else "D" if pts == 1 else "L")

    return TeamForm(
        team=team,
        form_ppg=float(np.mean(points)),
        goals_for_avg=float(np.mean(gf)),
        goals_against_avg=float(np.mean(ga)),
        last_results=results,
        form_season=season,
    )


def _rolling_attack_stats(team: str) -> tuple[float, float]:
    df = _load_matches()
    current = df[df["season"] == df["season"].max()]
    games = _team_matches(current, team).tail(FORM_WINDOW)
    if games.empty:
        return 10.0, 3.5
    shots = []
    sot = []
    for row in games.itertuples(index=False):
        is_home = row.home_team == team
        if is_home:
            shots.append(getattr(row, "home_shots", np.nan))
            sot.append(getattr(row, "home_shots_on_target", np.nan))
        else:
            shots.append(getattr(row, "away_shots", np.nan))
            sot.append(getattr(row, "away_shots_on_target", np.nan))
    return _avg(pd.Series(shots), 10.0), _avg(pd.Series(sot), 3.5)


def head_to_head(home_team: str, away_team: str, limit: int = 6) -> HeadToHeadSummary:
    df = _load_matches()
    mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team))
        | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )
    meetings = df.loc[mask].sort_values("date")
    home_wins = draws = away_wins = 0
    recent: list[dict] = []
    for row in meetings.itertuples(index=False):
        if row.home_team == home_team:
            if row.result == "H":
                home_wins += 1
            elif row.result == "D":
                draws += 1
            else:
                away_wins += 1
        else:
            if row.result == "A":
                home_wins += 1
            elif row.result == "D":
                draws += 1
            else:
                away_wins += 1
        recent.append(
            {
                "date": str(pd.Timestamp(row.date).date()),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "score": f"{row.home_goals}-{row.away_goals}",
                "result": row.result,
            }
        )
    recent = recent[-limit:]
    return HeadToHeadSummary(
        meetings=int(len(meetings)),
        home_wins=home_wins,
        draws=draws,
        away_wins=away_wins,
        recent=list(reversed(recent)),
    )


def preview_fixture(home_team: str, away_team: str) -> MatchPreviewResponse:
    teams = set(list_teams())
    all_teams = set(_load_matches()["home_team"]) | set(_load_matches()["away_team"])
    if home_team not in all_teams or away_team not in all_teams:
        raise ValueError("Unknown team selected")
    if home_team == away_team:
        raise ValueError("Home and away teams must be different")
    # allow selecting historical teams even if not in latest season picker
    _ = teams
    return MatchPreviewResponse(
        home=team_form(home_team),
        away=team_form(away_team),
        head_to_head=head_to_head(home_team, away_team),
    )


def feature_row(home_team: str, away_team: str) -> dict[str, float]:
    home = team_form(home_team)
    away = team_form(away_team)
    home_shots, home_sot = _rolling_attack_stats(home_team)
    away_shots, away_sot = _rolling_attack_stats(away_team)
    h2h = head_to_head(home_team, away_team, limit=50)
    if h2h.meetings == 0:
        h2h_home, h2h_draw = 0.4, 0.25
    else:
        h2h_home = h2h.home_wins / h2h.meetings
        h2h_draw = h2h.draws / h2h.meetings
    return {
        "home_form_ppg": home.form_ppg,
        "away_form_ppg": away.form_ppg,
        "home_gf_avg": home.goals_for_avg,
        "home_ga_avg": home.goals_against_avg,
        "away_gf_avg": away.goals_for_avg,
        "away_ga_avg": away.goals_against_avg,
        "home_shots_avg": home_shots,
        "away_shots_avg": away_shots,
        "home_sot_avg": home_sot,
        "away_sot_avg": away_sot,
        "h2h_home_win_rate": h2h_home,
        "h2h_draw_rate": h2h_draw,
    }
