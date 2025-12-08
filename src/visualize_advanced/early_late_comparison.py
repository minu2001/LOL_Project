# src/visualize_advanced/early_late_comparison.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 🌟 데이터 경로 설정
DATA_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\data\opscore_results.csv"
VISUALIZATION_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\visualizations"


def plot_early_late_comparison(df_score: pd.DataFrame, save=True):
    """
    각 플레이어의 Early Phase와 Late Phase 기여도 점수를 비교하는 산점도 시각화.
    """

    # 1. Early 및 Late Phase 데이터 필터링
    df_compare = df_score[df_score['phase'].isin(['early', 'late'])].copy()

    # 2. Match ID, Player ID, Lane 별 평균 final_score_norm 계산
    df_agg = df_compare.groupby(['match_id', 'pid', 'lane', 'phase'])['final_score_norm'].mean().reset_index()

    # 3. Early Score와 Late Score를 옆으로 펼침 (Pivot)
    df_pivot = df_agg.pivot_table(
        index=['match_id', 'pid', 'lane'],
        columns='phase',
        values='final_score_norm'
    ).reset_index().rename(columns={'early': 'Early_Score', 'late': 'Late_Score'})

    # NaN 값 제거 (Early 또는 Late 중 하나만 기록된 경우)
    df_pivot = df_pivot.dropna(subset=['Early_Score', 'Late_Score'])

    if df_pivot.empty:
        print("❌ 데이터셋에 Early 또는 Late Phase 기여도 점수 쌍이 없습니다.")
        return

    # 4. 시각화
    plt.figure(figsize=(10, 8))

    # Lane별 색상 구분하여 산점도 그리기
    sns.scatterplot(
        x='Early_Score',
        y='Late_Score',
        hue='lane',
        data=df_pivot,
        alpha=0.6,
        s=50
    )

    # 🌟 Y=X 대각선 (일관성 선)
    max_val = max(df_pivot['Early_Score'].max(), df_pivot['Late_Score'].max())
    min_val = min(df_pivot['Early_Score'].min(), df_pivot['Late_Score'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Consistent (Y=X)')

    plt.title("Player Contribution: Early Phase vs. Late Phase Score", fontsize=16)
    plt.xlabel("Early Phase Contribution Score (X)")
    plt.ylabel("Late Phase Contribution Score (Y)")
    plt.grid(alpha=0.3)
    plt.legend(title='Lane')
    plt.tight_layout()

    # 5. 저장
    if save:
        out_dir = os.path.join(VISUALIZATION_PATH, "phase_comparison")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "early_vs_late_score.png")
        plt.savefig(path, dpi=200)
        print(f"✔ Saved Early vs Late Score Plot: {path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "data", DATA_PATH)

        if not os.path.exists(data_path):
            print(f"❌ 데이터 파일 로드 실패: {DATA_PATH} 경로에 파일이 없습니다.")
            exit()

        df_test = pd.read_csv(data_path)
        print(f"✔️ {os.path.basename(data_path)} 파일 로드 성공. (Rows: {len(df_test)})")

        plot_early_late_comparison(df_test, save=True)

    except Exception as e:
        print(f"❌ 단독 실행 중 심각한 오류 발생: {e}")