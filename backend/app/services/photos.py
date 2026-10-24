import pandas as pd


def photo_url(row: pd.Series) -> str | None:
    code = row.get("photo_code") if hasattr(row, "get") else None
    if code is None or pd.isna(code):
        return None
    try:
        # Current Premier League site cutouts (premierleague25), refreshed for 2026/27.
        # The older /premierleague/photos/.../p{code}.png files stopped updating in 2024.
        return (
            "https://resources.premierleague.com/premierleague25/photos/players/110x140/"
            f"{int(code)}.png"
        )
    except (TypeError, ValueError):
        return None
