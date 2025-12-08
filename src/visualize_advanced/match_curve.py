# src/visualize_advanced/match_curve.py

import matplotlib

matplotlib.use('TkAgg')  # TkAgg 백엔드로 변경 (GUI 팝업 문제 해결)

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tempfile  # 안전한 임시 저장 경로를 위해 tempfile 모듈 사용

# VISUALIZATION_PATH 설정 (임시 파일 저장을 위한 안전한 경로로 변경)
VISUALIZATION_PATH = tempfile.gettempdir()  # 시스템의 임시 폴더 사용


def calculate_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """모델 예측 Baseline 대비 Ratio를 계산하여 DataFrame에 추가합니다."""

    df["model_baseline_score"] = df['late_model_score']
    df['end_phase_start'] = df['duration_min'] - 1

    df.loc[df['minute'] <= 15, "model_baseline_score"] = df['early_model_score']
    df.loc[df['minute'] >= df['end_phase_start'], "model_baseline_score"] = df['end_model_score']

    df["ratio_to_model_baseline"] = \
        df["final_score_norm"] / (df["model_baseline_score"].replace(0, 1e-6) + 1e-6)

    df = df.drop(columns=['end_phase_start'], errors='ignore')
    return df


def plot_match_curve(df_minute: pd.DataFrame, match_id: str, save=True):
    """
    특정 경기(match_id)의 시간축 라인별 기여도 곡선 시각화.
    """
    # ... (함수 본문은 이전 답변의 최종 코드를 그대로 사용) ...
    # (코드 길이를 위해 생략합니다. 로직은 이전 답변과 동일합니다.)
    # --------------------------------------------------------------------
    df_match = df_minute[df_minute["match_id"] == match_id].copy()

    if df_match.empty:
        print(f"[WARN] match_id={match_id} not found.")
        return

    required_cols = ['ratio_to_model_baseline', 'team_id']
    if not all(col in df_match.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_match.columns]
        print(f"❌ 오류: 시각화에 필요한 컬럼 ({', '.join(missing_cols)}) 중 일부가 DataFrame에 없습니다. Ratio 계산 단계를 확인하세요.")
        return

    lanes = ["TOP", "JUNGLE", "MIDDLE", "ADC", "SUPPORT"]
    teams = {100: "Blue", 200: "Red"}

    duration_min = df_match['duration_min'].max() if not df_match['duration_min'].empty else 30
    end_phase_start = duration_min - 1

    # 🌟🌟🌟 1. Baseline 재보정 (Median Normalization) 🌟🌟🌟
    median_ratio = df_minute["ratio_to_model_baseline"].median()
    calibration_factor = max(1e-6, median_ratio)

    df_match["ratio_final"] = df_match["ratio_to_model_baseline"] / calibration_factor

    # 🌟🌟🌟 2. 0분 시작점 1.0 보정 (공정한 출발점) 🌟🌟🌟
    if 0 in df_match['minute'].values:
        print(f"[INFO] 0분 데이터를 Baseline(1.0)으로 강제 보정합니다. (Baseline Factor: {calibration_factor:.3f})")
        df_match.loc[df_match['minute'] == 0, "ratio_final"] = 1.0

    plt.figure(figsize=(16, 10))

    # 🌟🌟🌟 모델 단계 영역 표시 🌟🌟🌟
    plt.axvspan(0, 15, color='green', alpha=0.1, label='Early Phase Model (0-15 min)')
    plt.axvspan(end_phase_start, duration_min + 1, color='purple', alpha=0.1, label='End Phase Model (End min)')

    # 🌟🌟🌟 1인분 기준선 표시 🌟🌟🌟
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                label='1.0 Baseline (Challenger Median)')

    # -----------------------------------------------------
    # 🌟🌟🌟 라인별 곡선 플롯 (EWMA 평활화 적용) 🌟🌟🌟
    # -----------------------------------------------------

    ewma_span = 15

    for team_id, team_name in teams.items():
        color = 'blue' if team_id == 100 else 'red'

        for lane in lanes:
            df_lane_team = df_match[
                (df_match["lane"] == lane) &
                (df_match["team_id"] == team_id)
                ].copy()

            if df_lane_team.empty:
                continue

            x = df_lane_team["minute"]
            y = df_lane_team["ratio_final"]

            y_smooth = y.ewm(span=ewma_span, min_periods=1, adjust=False).mean()

            label_name = f"{team_name} {lane}"
            plt.plot(x, y_smooth, label=label_name, linewidth=2, color=color,
                     linestyle='-' if lane not in ['ADC', 'SUPPORT'] else '--')

    # Event 표시 - 오브젝트
    if "event" in df_match.columns:
        df_obj = df_match[df_match["event"] == "objective"]
        for _, row in df_obj.iterrows():
            m = row["minute"]
            plt.axvline(x=m, color="gray", linestyle=":", alpha=0.3)
    else:
        pass

    # 🌟 타이틀 수정 (보정 정보 명시)
    alpha_value = 2 / (ewma_span + 1)
    plt.title(f"Match Contribution Curve (Median Calibrated, EWMA $\\alpha={alpha_value:.3f}$)\nMatch ID: {match_id}",
              fontsize=18)
    plt.xlabel("Minute")

    plt.ylabel(f"Performance Ratio (1.0 = Challenger Median, Baseline Factor: {calibration_factor:.3f})")

    plt.ylim(0, 4.0)

    # 범례 재설정
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h

    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', ncol=2)
    plt.grid(alpha=0.3)

    # 🌟🌟🌟 저장 경로를 시스템 임시 폴더로 변경 🌟🌟🌟
    if save:
        out_dir = os.path.join(VISUALIZATION_PATH, "match_curve")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"match_{match_id}_ratio_to_model_baseline_final.png")
        plt.savefig(out_path, dpi=200)
        print(f"[Saved] 결과 이미지가 다음 경로에 저장되었습니다: {out_path}")

    plt.show()


# --------------------------------------------------------------------

# 🌟 단독 실행 로직
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
                print("   이는 모델 예측 기준(Gold)과 평가 기준(XP)의 불일치를 유발합니다.")
                print("   정확한 분석을 위해 'normalization.py'와 'scoring.py' 실행을 확인하세요.")
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
            df_test['team_id'] = np.where(df_test['pid'] <= 5, 100, 200)

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

            # 🌟🌟🌟 save=True로 강제 저장 🌟🌟🌟
            plot_match_curve(df_test, match_id=test_match_id, save=True)
        else:
            print("❌ 'match_id' 컬럼이 없거나 비어 있어 시각화를 진행할 수 없습니다.")


    except Exception as e:
        print(f"❌ 단독 실행 중 심각한 오류 발생: {e}")