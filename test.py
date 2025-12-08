"""
main.py – 10개만 테스트하는 버전
"""

from src.load_data import (
    convert_json_to_parquet,
    get_parquet_paths
)

from src.feature_extract import extract_minute_features
from src.build_phase_datasets import build_phase_datasets
from src.model_training import train_all_models
from src.scoring import compute_opscore
from src.visualize import visualize_feature_importance, visualize_opscore_distribution


def main():

    print("📌 STEP 0) JSON → Parquet 변환")
    # 이미 변환했으면 주석 처리해도 됨
    convert_json_to_parquet()

    print("📌 STEP 1) Parquet 파일 목록 로딩")
    match_paths, timeline_paths = get_parquet_paths()
    print(f"   Matches: {len(match_paths)}, Timelines: {len(timeline_paths)}")

    # [핵심 수정] 10개만 잘라서 테스트!
    print("⚡ 테스트 모드: 10개 파일만 처리합니다.")
    test_match_paths = match_paths[:10]
    test_timeline_paths = timeline_paths[:10]

    print("📌 STEP 2) Minute-level Feature 생성 (Streaming)")
    # 잘린 리스트(test_...)를 넣습니다.
    df_minute = extract_minute_features(test_match_paths, test_timeline_paths)

    # 데이터가 비어있으면 중단
    if df_minute.empty:
        print("❌ 데이터 추출 실패. 종료합니다.")
        return

    print("📌 STEP 3) Early/Late/End Phase 데이터셋 분리")
    df_early, df_late, df_end = build_phase_datasets(df_minute)

    df_early.to_csv("data/phase_early.csv", index=False)
    df_late.to_csv("data/phase_late.csv", index=False)
    df_end.to_csv("data/phase_end.csv", index=False)

    print("✔ Phase datasets 저장 완료!")

    print("📌 STEP 4) 모델 학습 시작")
    train_all_models(df_early, df_late, df_end)

    print("📌 STEP 5) OPScore 계산")
    df_score = compute_opscore(df_minute)
    df_score.to_csv("data/opscore_results.csv", index=False)

    print("📌 STEP 6) 시각화 실행")
    visualize_feature_importance()
    visualize_opscore_distribution(df_score)

    print("🎉 테스트 파이프라인 완료!")


if __name__ == "__main__":
    main()