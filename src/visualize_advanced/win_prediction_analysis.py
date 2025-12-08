# src/visualize_advanced/win_prediction_analysis.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.font_manager as fm

# 🌟 데이터 경로 설정
DATA_PATH = "data/opscore_results.csv"
VISUALIZATION_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\visualizations"

# 🌟 한글 폰트 (Windows 100% 안정)
KOREAN_FONT_NAME = 'Malgun Gothic'
plt.rcParams['font.family'] = KOREAN_FONT_NAME
plt.rcParams['axes.unicode_minus'] = False


# =======================================================
# [1] 데이터 준비 및 승패 예측 모델 학습
# =======================================================

def prepare_data_and_predict():
    print("🚀 STEP 7: 기여도 점수 기반 승패 예측 모델링 시작...")

    # 🚨 파일 로드 (opscore_results.csv 사용)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, DATA_PATH)

    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일 로드 실패: {DATA_PATH} 경로에 파일이 없습니다.")
        return None

    df_score = pd.read_csv(data_path)

    # 🌟🌟🌟 오류 해결 로직: 'win' 컬럼이 없을 경우 재정의 🌟🌟🌟
    if 'win' not in df_score.columns or df_score['win'].isnull().all():
        print("⚠️ 'win' 컬럼이 없어, 'target_gold' 최대값을 기준으로 승패를 임시 생성합니다.")

        # 1. 각 매치별 최종 누적 골드 (Max target_gold)를 가진 팀 찾기
        df_score['max_gold_in_match'] = df_score.groupby('match_id')['target_gold'].transform('max')

        # 2. 해당 매치에서 Max Gold를 가진 팀을 승리(True)로 설정
        df_score['win'] = df_score['target_gold'] == df_score['max_gold_in_match']
    # 🌟🌟🌟

    # 1. 각 매치/팀별 최종 기여도 점수 평균 및 승패 상태 집계
    df_match_summary = df_score.groupby(['match_id', 'team_id']).agg(
        avg_score=('final_score_norm', 'mean'),
        win=('win', 'first')
    ).reset_index()

    # 2. Match_ID를 기준으로 Team 100(Blue)과 Team 200(Red)의 점수를 옆으로 펼침
    df_pivot = df_match_summary.pivot(
        index='match_id',
        columns='team_id',
        values=['avg_score', 'win']
    )

    # 컬럼 정리
    df_pivot.columns = ['_'.join(map(str, col)).strip() for col in df_pivot.columns.values]
    df_pivot = df_pivot.reset_index()

    # 승패 컬럼 설정 (Team 100의 승패를 기준으로 설정)
    df_pivot['target_win'] = df_pivot['win_100'].astype(int)

    # 🚨 최종 피처 X: Team 100과 Team 200의 평균 기여도 점수만 사용
    X = df_pivot[['avg_score_100', 'avg_score_200']]
    y = df_pivot['target_win']

    # 3. 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"✔️ Logistic Regression 모델 학습 완료.")
    print(f"   -> 테스트 정확도 (Accuracy): {accuracy:.4f}")

    return y_test, y_pred, y_pred_proba, accuracy


# =======================================================
# [2] ROC Curve 시각화
# =======================================================

def plot_win_prediction_metrics(y_test, y_pred_proba, accuracy, save=True):
    """ROC Curve와 Confusion Matrix를 시각화"""

    # 1. ROC Curve 계산
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # 2. 시각화 (2x1 Subplots)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # A. ROC Curve
    ax[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax[0].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate (FPR)')
    ax[0].set_ylabel('True Positive Rate (TPR)')
    ax[0].set_title('A. ROC Curve for Win Prediction', fontsize=14)
    ax[0].legend(loc="lower right")

    # B. Confusion Matrix (정확도 대신 예측 경향성 확인)
    cm = confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
    ax[1].set_xlabel('Predicted Label')
    ax[1].set_ylabel('True Label')
    ax[1].set_title(f'B. Confusion Matrix (Accuracy: {accuracy:.4f})', fontsize=14)
    ax[1].xaxis.set_ticklabels(['Loss (200 Win)', 'Win (100 Win)'])
    ax[1].yaxis.set_ticklabels(['Loss (200 Win)', 'Win (100 Win)'])

    fig.suptitle("Win Prediction Analysis using Final Contribution Score", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 3. 저장
    if save:
        out_dir = os.path.join(VISUALIZATION_PATH, "win_prediction")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "win_prediction_analysis.png")
        plt.savefig(path, dpi=200)
        print(f"✔ Saved Win Prediction Analysis: {path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        results = prepare_data_and_predict()
        if results is not None:
            y_test, y_pred, y_pred_proba, accuracy = results
            plot_win_prediction_metrics(y_test, y_pred_proba, accuracy, save=True)
            print("🎉 승패 예측 시각화 완료!")
    except Exception as e:
        print(f"❌ 최종 시각화 실행 중 오류 발생: {e}")