from src.load_data import convert_json_to_parquet, get_parquet_paths
from src.feature_extract import extract_minute_features
from src.build_phase_datasets import build_phase_datasets
from src.model_training import train_all_models
from src.scoring import compute_opscore
from src.visualize import visualize_feature_importance, visualize_opscore_distribution


def main():

    print("📌 STEP 0) JSON → Parquet 변환")
    convert_json_to_parquet()

    print("📌 STEP 1) 파일 경로 로드")
    match_paths, timeline_paths = get_parquet_paths()
    print(f"Matches = {len(match_paths)} | Timelines = {len(timeline_paths)}")

    print("📌 STEP 2) Minute Feature 추출")
    df_minute = extract_minute_features(match_paths, timeline_paths)

    print("📌 STEP 3) Phase Split")
    df_early, df_late, df_end = build_phase_datasets(df_minute)

    df_early.to_csv("data/phase_early.csv", index=False)
    df_late.to_csv("data/phase_late.csv", index=False)
    df_end.to_csv("data/phase_end.csv", index=False)

    print("📌 STEP 4) 모델 학습")
    train_all_models(df_early, df_late, df_end)

    print("📌 STEP 5) OPScore 계산")
    df_score                                                   = compute_opscore(df_minute)
    df_score.to_csv("data/opscore_results.csv", index=False)

    print("📌 STEP 6) 시각화")
    visualize_feature_importance()
    visualize_opscore_distribution(df_score)

    print("🎉 전체 파이프라인 완료!")


if __name__ == "__main__":
    main()
