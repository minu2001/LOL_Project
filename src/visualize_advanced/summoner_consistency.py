# src/visualize_advanced/summoner_consistency.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 🌟 VISUALIZATION_PATH를 절대 경로로 강제 지정
VISUALIZATION_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\visualizations"


def compute_stability_metrics(scores: list):
    """경기별 score 리스트를 기반으로 기복(stability) 분석 지표 계산."""
    arr = np.array(scores)

    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "cv": np.std(arr) / (np.mean(arr) + 1e-6),  # 변동계수
    }


def plot_summoner_consistency(df_minute: pd.DataFrame, puuid: str, save=True):
    """
    특정 소환사(puuid)의 경기별 기여도 변화 분석.
    """

    df_p = df_minute[df_minute["puuid"] == puuid].copy()

    if df_p.empty:
        print(f"[WARN] puuid={puuid} not found.")
        return

    # 1) 경기 단위로 평균 기여도 계산 및 1.0 중앙값 보정
    df_game = (
        df_p.groupby("match_id")["final_score_norm"]
        .mean()
        .reset_index()
        .rename(columns={"final_score_norm": "avg_score"})
    )

    # 🌟🌟🌟 Median=1.0 보정 🌟🌟🌟
    median_score = df_game["avg_score"].median()
    calibration_factor = max(1e-6, median_score)
    df_game["avg_score_calibrated"] = df_game["avg_score"] / calibration_factor

    # 경기 번호 부여 및 승/패 정보 매핑
    df_game["game_index"] = range(1, len(df_game) + 1)

    colors = 'gray'
    if "win" in df_p.columns:
        win_map = df_p.groupby("match_id")["win"].first()
        df_game["win"] = df_game["match_id"].map(win_map)
        colors = df_game["win"].map({True: "blue", False: "red"})

    # 2) Stability Metrics 계산
    stability = compute_stability_metrics(df_game["avg_score_calibrated"].tolist())

    print("===== Stability Metrics =====")
    print(f"Calibration Factor (Raw Median): {median_score:.3f}")
    print(stability)
    print("=============================")

    # 3) 시각화
    plt.figure(figsize=(16, 8))

    # 산점도 (승/패 색상 구분)
    plt.scatter(
        df_game["game_index"],
        df_game["avg_score_calibrated"],
        c=colors,
        alpha=0.7,
        label="Game Performance"
    )

    # 1.0 Baseline 표시 (중앙값)
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f"Median Performance (1.0)")

    # Rolling Mean
    rolling = df_game["avg_score_calibrated"].rolling(window=5, min_periods=1).mean()
    plt.plot(df_game["game_index"], rolling, color="black", linewidth=2, label="Rolling Mean (5 games)")

    plt.title(f"Summoner Consistency Analysis (Median=1.0)\nPUUID: {puuid}", fontsize=18)
    plt.xlabel("Game Index")
    plt.ylabel("Average Contribution Ratio (Median=1.0)")
    plt.grid(alpha=0.3)

    if "win" in df_p.columns:
        plt.legend(loc='upper right', handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Win', markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Loss', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Rolling Mean (5 games)'),
            plt.Line2D([0], [0], color='gray', linestyle='--', label='Median Performance (1.0)')
        ])
    else:
        plt.legend(loc='upper right')

    if save:
        out_dir = os.path.join(VISUALIZATION_PATH, "consistency")
        os.makedirs(out_dir, exist_ok=True)

        save_path = os.path.join(out_dir, f"{puuid}_consistency.png")
        plt.savefig(save_path, dpi=200)
        print(f"[Saved] 결과 이미지가 다음 경로에 저장되었습니다: {save_path}")

    plt.show()
    plt.close()

    return stability


# 🌟 단독 실행 로직 (동적으로 가장 적합한 PUUID를 찾습니다.)
if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "data", "minute_features.csv")

        if not os.path.exists(data_path):
            print(f"❌ 데이터 파일 로드 실패: {data_path} 경로에 파일이 없습니다. 메인 파이프라인을 재실행하세요.")
            exit()

        df_test = pd.read_csv(data_path)

        # 🌟🌟🌟 1. final_score_norm 임시 생성 (경고 방지) 🌟🌟🌟
        if 'final_score_norm' not in df_test.columns:
            source_col = 'target_gold' if 'target_gold' in df_test.columns else 'xp'
            if source_col in df_test.columns:
                print(f"⚠️ 'final_score_norm'이 없어 '{source_col}'를 기반으로 임시 생성합니다.")
                df_test['final_score_norm'] = (df_test[source_col] - df_test[source_col].min()) / (
                        df_test[source_col].max() - df_test[source_col].min() + 1e-6)
            else:
                print("❌ final_score_norm 생성을 위한 필수 컬럼(target_gold/xp)이 없습니다.")
                exit()

        # 🌟🌟🌟 2. puuid 컬럼 생성 및 최적 PUUID 선정 🌟🌟🌟
        if 'puuid' not in df_test.columns:
            if 'pid' in df_test.columns:
                df_test['puuid'] = df_test['pid'].astype(str) + "_" + df_test['match_id'].astype(str)
            else:
                print("❌ 'puuid' 생성을 위한 필수 컬럼('pid')이 없습니다.")
                exit()

        if 'win' not in df_test.columns and 'team_id' in df_test.columns:
            df_test['win'] = df_test['team_id'] == 100

            # 🚨 최적 PUUID 동적 선정 로직
        puuid_match_counts = df_test.groupby('puuid')['match_id'].nunique()
        frequent_puuids = puuid_match_counts[puuid_match_counts >= 2]

        if frequent_puuids.empty:
            test_puuid = puuid_match_counts.idxmax()
            print(f"⚠️ 2경기 이상 플레이한 소환사 없음. 가장 많은 경기의 소환사를 사용합니다: {test_puuid}")
        else:
            test_puuid = frequent_puuids.idxmax()
            print(f"✅ 일관성 분석을 위해 {frequent_puuids.max()}경기 플레이한 소환사를 선정합니다: {test_puuid}")

        print(f"✔️ 파일 로드 성공. 시각화 시작. (PUUID: {test_puuid})")

        plot_summoner_consistency(df_test, puuid=test_puuid, save=True)

    except Exception as e:
        print(f"❌ 단독 실행 중 심각한 오류 발생: {e}")