import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

sns.set(style="whitegrid")

# 🌟 절대 경로 설정 (저장 경로를 명확히 함)
BASE_DIR = r"C:\Users\user\PycharmProjects\Last_LOL_Project"
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALIZATION_PATH = os.path.join(BASE_DIR, "visualizations", "distribution")

FEATURES = [
    "cs_per_min",
    "xp_per_min",
    "dpm",  # <-- DPM 포함
    "kills_per_min",
    "assists_per_min",
    "deaths_per_min",
    "vision_score_per_min",
    "jung_cs_per_min",
    "damage_taken_per_min",
]


def load_phase_data():
    # 데이터 로드 (절대 경로 사용)
    try:
        df_early = pd.read_csv(os.path.join(DATA_DIR, "phase_early.csv"))
        df_late = pd.read_csv(os.path.join(DATA_DIR, "phase_late.csv"))
        df_end = pd.read_csv(os.path.join(DATA_DIR, "phase_end.csv"))
    except FileNotFoundError:
        print(f"❌ 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {DATA_DIR}")
        return pd.DataFrame()

    df_early["phase"] = "Early"
    df_late["phase"] = "Late"
    df_end["phase"] = "End"

    return pd.concat([df_early, df_late, df_end])


def plot_feature_distribution():
    # 폴더가 없으면 생성
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    print(f"📂 저장 경로 확인: {VISUALIZATION_PATH}")

    df = load_phase_data()
    if df.empty:
        return

    # 🌟🌟🌟 FIX 1: 아웃라이어 제거를 위한 Y축 제한 값 계산 (99.5% 분위수) 🌟🌟🌟
    y_limits = {}
    for feat in FEATURES:
        if feat in df.columns:
            # 99.5% 분위수를 Y축 상한으로 설정
            y_limits[feat] = df[feat].quantile(0.995)
    # 🌟🌟🌟

    lanes = df["lane"].unique()
    phases = ["Early", "Late", "End"]

    print("🚀 분포(Boxplot/KDE) 시각화 생성 시작...")

    for lane in lanes:
        lane_df = df[df["lane"] == lane]

        for feat in FEATURES:
            if feat not in df.columns:
                continue

            y_limit = y_limits.get(feat, df[feat].max())

            # 1. Boxplot (FutureWarning 및 Blank Plot 해결)
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=lane_df,
                x="phase",
                y=feat,
                hue="phase",  # FIX: FutureWarning 해결
                palette="Set2",
                order=phases,
                legend=False  # FIX: FutureWarning 해결
            )
            plt.title(f"{lane} – {feat} Distribution by Phase (Y-axis Capped at 99.5th %)")
            plt.ylabel(feat)

            # Y축 제한 적용
            if y_limit > 0 and y_limit < df[feat].max():
                plt.ylim(0, y_limit)

            plt.tight_layout()
            save_path = os.path.join(VISUALIZATION_PATH, f"{lane}_{feat}_boxplot.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

            # 2. KDE Plot (0 variance 경고는 그대로 발생하지만, 유효한 데이터는 그림)
            plt.figure(figsize=(10, 6))
            valid_plot = False
            for p in phases:
                sub = lane_df[lane_df["phase"] == p][feat]
                # 데이터가 있고 분산이 0이 아닐 때만 그림 (nunique > 1)
                if len(sub) > 0 and sub.nunique() > 1:
                    sns.kdeplot(sub, fill=True, alpha=0.35, label=p)
                    valid_plot = True

            if valid_plot:
                plt.title(f"{lane} – {feat} KDE by Phase")
                plt.xlabel(feat)
                plt.ylabel("Density")
                plt.legend()
                plt.tight_layout()
                save_path_kde = os.path.join(VISUALIZATION_PATH, f"{lane}_{feat}_kde.png")
                plt.savefig(save_path_kde, dpi=300)

            plt.close()

        print(f"   -> {lane} 라인 시각화 완료")

    print(f"✅ 저장 완료! 폴더를 확인하세요: {VISUALIZATION_PATH}")


if __name__ == "__main__":
    plot_feature_distribution()