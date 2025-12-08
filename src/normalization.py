import os
import pandas as pd
import numpy as np

from .config import DATA_DIR


# 정규화 대상에서 항상 제외할 메타 컬럼들
META_COLS = [
    "match_id", "gameId", "gameId_str",
    "participantId", "puuid",
    "teamId", "teamPosition", "lane", "role",
    "support_role",
    "championName", "summonerName",
    "win",
    "gameDuration", "minute", "phase"
]

# 기본 파일 경로
MINUTE_FEATURES_FILE = os.path.join(DATA_DIR, "minute_features.csv")
NORMALIZED_MINUTE_FILE = os.path.join(DATA_DIR, "minute_features_normalized.csv")
NORM_STATS_FILE = os.path.join(DATA_DIR, "norm_stats.csv")


def load_minute_features(path: str = MINUTE_FEATURES_FILE) -> pd.DataFrame:
    """
    분 단위 피처 데이터 로드.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[normalization] minute_features 파일이 없음: {path}")
    df = pd.read_csv(path)
    return df


def get_feature_cols(df: pd.DataFrame):
    """
    메타 컬럼을 제외한 수치형 피처 컬럼 리스트 반환.
    """
    feature_cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def compute_medians(df: pd.DataFrame, feature_cols):
    """
    각 피처의 전체 분포 기준 median을 계산.
    median = 0 인 경우는 1e-6으로 대체해서 나누기 에러 방지.
    """
    med = df[feature_cols].median()
    med = med.replace(0, 1e-6)
    return med


def apply_normalization(df: pd.DataFrame, medians: pd.Series, feature_cols, prefix: str = "norm_"):
    """
    각 피처를 median으로 나눠서 정규화.
    norm_x = x / median(x)
    """
    df = df.copy()
    for col in feature_cols:
        norm_col = prefix + col
        df[norm_col] = df[col] / medians[col]

        # NaN, inf 방지
        df[norm_col] = df[norm_col].replace([np.inf, -np.inf], np.nan)
        df[norm_col] = df[norm_col].fillna(0.0)

    return df


def save_norm_stats(medians: pd.Series, path: str = NORM_STATS_FILE):
    """
    median 값들을 CSV로 저장해서 나중에 재사용 가능하게 함.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    med_df = medians.to_frame(name="median")
    med_df.index.name = "feature"
    med_df.to_csv(path, encoding="utf-8-sig")


def normalize_minute_features(
    input_path: str = MINUTE_FEATURES_FILE,
    output_path: str = NORMALIZED_MINUTE_FILE,
    stats_path: str = NORM_STATS_FILE,
):
    """
    메인 진입점:
    - minute_features.csv 로드
    - 수치형 피처들 median 계산
    - 정규화된 피처(norm_*) 생성
    - 결과 및 median 통계 저장
    """
    print(f"[normalization] Load minute features from: {input_path}")
    df = load_minute_features(input_path)

    feature_cols = get_feature_cols(df)
    print(f"[normalization] numeric feature cols: {len(feature_cols)}개")
    if not feature_cols:
        raise ValueError("[normalization] 수치형 피처가 없음. feature_extract 단계 확인 필요")

    medians = compute_medians(df, feature_cols)
    print("[normalization] median 계산 완료")

    df_norm = apply_normalization(df, medians, feature_cols, prefix="norm_")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_norm.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[normalization] normalized 파일 저장: {output_path}")

    save_norm_stats(medians, stats_path)
    print(f"[normalization] median 통계 저장: {stats_path}")

    return df_norm, medians


if __name__ == "__main__":
    # 단독 실행용
    normalize_minute_features()
