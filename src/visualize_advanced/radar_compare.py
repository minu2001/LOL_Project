# src/visualize_advanced/plot_feature_importance.py

import os
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
import seaborn as sns
import matplotlib



# 🌟 VISUALIZATION_PATH를 절대 경로로 강제 지정
VISUALIZATION_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\visualizations"
MODEL_DIR = "models"

# 🌟 프로젝트 모델 구조 정의 (반복 실행을 위한 정의)
MODEL_STRUCTURE = {
    "TOP": ["early", "late", "end"],
    "JUNGLE": ["early", "late", "end"],
    "MID": ["early", "late", "end"],
    "ADC": ["early", "late", "end"],
    "SUPPORT": ["Damage", "Enchanter", "Tank", "Assassin"],  # SUPPORT 역할군 정의
}


def plot_feature_importance_for_model(lane, phase, role=None, save=True, top_n=10):
    """
    특정 라인/페이즈/역할 모델의 피처 중요도를 시각화합니다.
    """

    # 1. 모델 경로 설정
    if lane == "SUPPORT" and role:
        # SUPPORT_Damage_early.cbm
        model_name = f"{lane}_{role}_{phase}.cbm"
    elif lane != "SUPPORT":
        # TOP_early.cbm
        model_name = f"{lane}_{phase}.cbm"
    else:
        return  # Skip incomplete SUPPORT definitions

    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        return

    try:
        # 2. 모델 로드 및 중요도 추출
        model = CatBoostRegressor()
        # 주의: CatBoost는 학습 파라미터를 로드해야 하므로, 모델 파일에서 바로 로드합니다.
        model.load_model(model_path)

        importance = model.get_feature_importance()
        features = model.feature_names_

        df_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        df_importance = df_importance.sort_values(by='Importance', ascending=False)

        # 3. 상위 N개 피처 선택
        df_plot = df_importance.head(top_n)

        # 4. 시각화
        plt.figure(figsize=(10, 6))

        # 중요도가 0인 피처는 제거하고 시각화합니다.
        df_plot = df_plot[df_plot['Importance'] > 0]

        if df_plot.empty:
            print(f"⚠️ {model_name}: All feature importances are zero or near zero.")
            return

        # Bar plot 생성 (수평 막대 그래프)
        sns.barplot(x='Importance', y='Feature', data=df_plot, palette='viridis')

        # 타이틀 설정
        if lane == "SUPPORT" and role:
            plot_title = f"Feature Importance: {lane} ({role}) - {phase.capitalize()}"
        else:
            plot_title = f"Feature Importance: {lane} - {phase.capitalize()}"

        plt.title(plot_title, fontsize=14)
        plt.xlabel("Importance Score (Higher = More Impact)")
        plt.ylabel("Feature")
        plt.tight_layout()

        # 5. 저장
        if save:
            out_dir = os.path.join(VISUALIZATION_PATH, "feature_importance")
            os.makedirs(out_dir, exist_ok=True)

            # 파일 이름은 models/ 폴더의 구조를 따릅니다.
            save_name = model_name.replace(".cbm", ".png")
            path = os.path.join(out_dir, save_name)
            plt.savefig(path, dpi=200)
            print(f"✔ Saved Feature Importance: {path}")

        plt.close()  # 메모리 해제

    except Exception as e:
        print(f"❌ Error processing model {model_name}: {e}")


def run_all_feature_importance_plots(save=True):
    print("=========================================")
    print("📊 Generating All Feature Importance Plots")
    print("=========================================")

    for lane in MODEL_STRUCTURE:
        for phase_or_role in MODEL_STRUCTURE[lane]:
            if lane == "SUPPORT" and phase_or_role not in ["early", "late", "end"]:
                # Support Roles (Damage, Enchanter, Tank, Assassin)
                role = phase_or_role
                for sup_phase in ["early", "late", "end"]:
                    plot_feature_importance_for_model(lane, sup_phase, role=role, save=save)
            elif lane != "SUPPORT":
                # TOP, MID, JUNGLE, ADC (Phase: early, late, end)
                phase = phase_or_role
                plot_feature_importance_for_model(lane, phase, save=save)

    print("🎉 Feature Importance Plots Generation Complete.")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")  # 테마 설정
    run_all_feature_importance_plots(save=True)