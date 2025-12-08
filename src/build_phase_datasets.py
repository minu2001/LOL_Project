# src/build_phase_datasets.py

import pandas as pd

def build_phase_datasets(df):
    """
    Early: minute < 15
    Late:  15 <= minute < end_minute
    End:   minute == end_minute
    """

    if "match_id" not in df.columns:
        raise KeyError(f"❌ match_id 없음. columns = {df.columns.tolist()}")

    # end_minute = 게임 종료 시점 (이미 duration_min에 들어있음)
    df["end_minute"] = df["duration_min"]

    df_early = df[df["minute"] < 15].copy()

    df_late = df[
        (df["minute"] >= 15) &
        (df["minute"] < df["end_minute"])
    ].copy()

    # end = 딱 게임 종료 시점
    df_end = df[df["minute"] == df["end_minute"]].copy()

    return df_early, df_late, df_end
