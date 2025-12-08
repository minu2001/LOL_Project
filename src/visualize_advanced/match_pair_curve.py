# src/visualize_advanced/match_pair_curve.py

import matplotlib

matplotlib.use('TkAgg')

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 🌟🌟🌟 수정: VISUALIZATION_PATH를 절대 경로로 강제 지정 🌟🌟🌟
VISUALIZATION_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\visualizations"


def calculate_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """모델 예측 Baseline 대비 Ratio를 계산하여 DataFrame에 추가합니다."""

    # 필수 컬럼 검사 (단독 실행 시 __main__에서 처리됨)
    df["model_baseline_score"] = df['late_model_score']
    df['end_phase_start'] = df['duration_min'] - 1

    df.loc[df['minute'] <= 15, "model_baseline_score"] = df['early_model_score']
    df.loc[df['minute'] >= df['end_phase_start'], "model_baseline_score"] = df['end_model_score']

    df["ratio_to_model_baseline"] = \
        df["final_score_norm"] / (df["model_baseline_score"].replace(0, 1e-6) + 1e-6)

    df = df.drop(columns=['end_phase_start'], errors='ignore')
    return df


def plot_match_pair_curve(df_minute: pd.DataFrame, match_id: str, save=True):
    """
    특정 경기의 라인별(TOP, JUNGLE 등) 기여도 추이를 5개의 분리된 플롯으로 시각화합니다.
    각 플롯에는 Actual과 Predicted (Baseline 1.0) 4개 곡선이 표시됩니다.
    """

    df_match = df_minute[df_minute["match_id"] == match_id].copy()

    if df_match.empty:
        print(f"[WARN] match_id={match_id} not found.")
        return

    # 필수 컬럼 검사 (Ratio는 이미 calculate_ratio에서 생성됨)
    required_cols = ['ratio_to_model_baseline', 'team_id']
    if not all(col in df_match.columns for col in required_cols):
        print("❌ 오류: Ratio 계산이 선행되지 않았거나 필수 컬럼이 누락되었습니다.")
        return

    lanes = ["TOP", "JUNGLE", "MIDDLE", "ADC", "SUPPORT"]
    teams = {100: "Blue", 200: "Red"}

    duration_min = df_match['duration_min'].max() if not df_match['duration_min'].empty else 30
    end_phase_start = duration_min - 1

    # 🌟🌟🌟 1. Baseline 재보정 및 0분 보정 🌟🌟🌟
    median_ratio = df_minute["ratio_to_model_baseline"].median()
    calibration_factor = max(1e-6, median_ratio)

    df_match["ratio_final"] = df_match["ratio_to_model_baseline"] / calibration_factor

    if 0 in df_match['minute'].values:
        print(f"[INFO] 0분 데이터를 Baseline(1.0)으로 강제 보정합니다. (Baseline Factor: {calibration_factor:.3f})")
        df_match.loc[df_match['minute'] == 0, "ratio_final"] = 1.0

    # 🌟🌟🌟 2. 5개의 플롯을 담을 Figure 생성 🌟🌟🌟
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
    axes = axes.flatten()

    ewma_span = 15  # 평활화 강도

    for i, lane in enumerate(lanes):
        ax = axes[i]  # 현재 라인의 플롯 지정

        # 🌟🌟🌟 1. Actual (실제 기여도) 곡선 🌟🌟🌟
        for team_id, team_name in teams.items():
            df_lane_team = df_match[
                (df_match["lane"] == lane) &
                (df_match["team_id"] == team_id)
                ].copy()

            if df_lane_team.empty: continue

            x = df_lane_team["minute"]
            y = df_lane_team["ratio_final"]  # 보정된 실제 기여도 Ratio

            # EWMA 평활화
            y_smooth = y.ewm(span=ewma_span, min_periods=1, adjust=False).mean()
            color = 'blue' if team_id == 100 else 'red'

            # 🌟 Actual 곡선 플롯 (실선)
            ax.plot(x, y_smooth, label=f"{team_name} Actual", linewidth=2, color=color, linestyle='-')

            # 🌟🌟🌟 2. Predicted (모델 기대치) 곡선 🌟🌟🌟
            # 모델 예측은 이미 1.0 Baseline으로 보정된 기대치를 나타내므로, 1.0 선을 Predicted로 사용
            ax.axhline(y=1.0, color=color, linestyle='--', linewidth=1, alpha=0.7, label=f"{team_name} Predicted")

        # 🌟🌟🌟 플롯 설정 🌟🌟🌟
        ax.set_title(f"Lane: {lane}", fontsize=16)
        ax.set_xlabel("Minute")
        ax.set_ylabel("Performance Ratio (1.0 = Expectation)")
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, 3.5)  # Y축 확장

        # 모델 단계 영역 표시 (모든 플롯에 공통)
        ax.axvspan(0, 15, color='green', alpha=0.05)  # Early Phase
        ax.axvspan(end_phase_start, duration_min + 1, color='purple', alpha=0.05)  # End Phase

        ax.legend(loc='upper left', fontsize=8)

    # 🌟🌟🌟 3. 최종 Figure 설정 🌟🌟🌟
    # 마지막 빈 플롯 제거
    fig.delaxes(axes[-1])

    fig.suptitle(f"Match ID: {match_id} - Lane Contribution Comparison", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Super title 공간 확보

    if save:
        out_dir = os.path.join(VISUALIZATION_PATH, "match_pair_curve")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"match_{match_id}_pair_comparison_final.png")
        plt.savefig(out_path, dpi=200)
        print(f"[Saved] 결과 이미지가 다음 경로에 저장되었습니다: {out_path}")

    plt.show()


# --------------------------------------------------------------------

# 🌟 단독 실행 로직 (Ratio 계산을 외부화)
if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "data", "minute_features.csv")

        if not os.path.exists(data_path):
            print(f"❌ 데이터 파일 로드 실패: {data_path} 경로에 파일이 없습니다.")
            exit()

        df_test = pd.read_csv(data_path)

        # 🌟🌟🌟 final_score_norm 누락 시 XP 기반으로 생성 (분석적 경고 필수) 🌟🌟🌟
        if 'final_score_norm' not in df_test.columns:
            if 'xp' in df_test.columns:
                print("\n\n###################################################################")
                print("🚨 WARNING: 분석적 오류 위험! (XP 임시 생성)")
                print("   'final_score_norm'이 없어 'xp'를 기반으로 임시 생성합니다.")
                print("###################################################################\n")

                min_xp = df_test['xp'].min()
                max_xp = df_test['xp'].max()
                df_test['final_score_norm'] = (df_test['xp'] - min_xp) / (max_xp - min_xp + 1e-6)
            else:
                print("❌ 'final_score_norm'과 'xp' 컬럼이 모두 없어 시각화를 진행할 수 없습니다.")
                exit()

        # duration_min 및 team_id 컬럼 검사 및 생성
        if 'duration_min' not in df_test.columns:
            df_test['duration_min'] = 30

        if 'team_id' not in df_test.columns:
            if 'pid' in df_test.columns:
                df_test['team_id'] = np.where(df_test['pid'] <= 5, 100, 200)
            else:
                print("❌ 'team_id' 생성을 위한 필수 컬럼('pid')이 없어 시각화를 진행할 수 없습니다.")
                exit()

        # 🌟🌟🌟 모델 예측 점수가 없으면 임시로 생성 🌟🌟🌟
        if 'early_model_score' not in df_test.columns:
            print("[WARN] 모델 예측 점수가 없어 final_score_norm 기반으로 임시 생성합니다.")
            df_test['early_model_score'] = df_test['final_score_norm'].rolling(window=15, min_periods=1).mean() * 1.1
            df_test['late_model_score'] = df_test['final_score_norm'].rolling(window=10, min_periods=1).mean() * 1.1
            df_test['end_model_score'] = df_test['late_model_score'] * 1.2

        # Ratio 계산 함수를 호출하여 컬럼을 추가
        df_test = calculate_ratio(df_test)

        if 'match_id' in df_test.columns and not df_test['match_id'].empty:
            test_match_id = df_test['match_id'].iloc[0]

            print(f"✔️ 파일 로드 성공. 시각화 시작. (Match ID: {test_match_id})")

            plot_match_pair_curve(df_test, match_id=test_match_id, save=True)
        else:
            print("❌ 'match_id' 컬럼이 없거나 비어 있어 시각화를 진행할 수 없습니다.")


    except Exception as e:
        print(f"❌ 단독 실행 중 심각한 오류 발생: {e}")