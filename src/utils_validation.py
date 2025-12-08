"""
utils_validation.py
데이터가 모델 학습 및 점수 계산에 사용되기 전에
이상치, 결측치, 라벨링 오류 등을 검사하는 모듈
"""

import numpy as np


def validate_minute_level_data(df):
    errors = []
    warnings = []

    # 1) lane 누락
    if df["lane"].isna().sum() > 0:
        errors.append("❌ lane 값이 없는 row가 있습니다.")

    # 2) end phase 체크
    grouped = df.groupby(["game_id", "lane"])
    for (gid, lane), subdf in grouped:
        end_count = subdf["is_end"].sum()
        if end_count != 1:
            warnings.append(
                f"⚠ game_id={gid}, lane={lane}: end phase가 {end_count}개입니다."
            )

    # 3) null 값 체크
    null_count = df.isna().sum().sum()
    if null_count > 0:
        errors.append(f"❌ Null 값 발견: 총 {null_count}개")

    # 4) Infinite 값 체크
    if np.isinf(df.select_dtypes(include=[float])).any().any():
        errors.append("❌ infinite 값이 포함되어 있습니다.")

    # 5) minute 순서 체크
    if df.sort_values(["game_id", "minute"]).index.equals(df.index) is False:
        warnings.append("⚠ minute 순서가 정렬되어 있지 않습니다.")

    is_valid = len(errors) == 0

    return is_valid, errors, warnings


def validate_before_visualization(df):
    issues = []

    if "opscore" not in df.columns:
        issues.append("❌ OPScore가 없는 데이터입니다.")

    if df["opscore"].isna().sum() > 0:
        issues.append("❌ OPScore 중 null 값이 있습니다.")

    if df["opscore"].max() - df["opscore"].min() < 0.1:
        issues.append("⚠ OPScore 값의 편차가 너무 적습니다. 모델이 의미 있게 학습되지 않았을 수 있습니다.")

    return issues
