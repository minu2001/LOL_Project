# src/visualize_advanced/cluster_feature_heatmap.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 🌟 데이터 경로 및 설정 (lol_final_excel_dataset 사용)
DATA_FILE_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\data\lol_final_excel_dataset - 복사본.csv"
VISUALIZATION_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\visualizations"
CLUSTER_N = 4  # 4개 군집 사용


def prepare_data_and_cluster():
    """lol4.py 로직을 단순화하여 클러스터링을 수행합니다."""

    if not os.path.exists(DATA_FILE_PATH):
        print(f"❌ 데이터 파일 로드 실패: {DATA_FILE_PATH} 경로에 파일이 없습니다.")
        return None, None

    df = pd.read_csv(DATA_FILE_PATH)

    # 🚨 군집 분석에 사용할 핵심 피처 (lol4.py와 동일하게 유지)
    feature_prefixes = ['TOP', 'JUNGLE', 'MIDDLE', 'Adc', 'Sup']
    feature_cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in feature_prefixes)]

    # 🚨 Match_ID, Team_ID를 포함하여 긴 형태로 변환하는 대신, 이 파일의 장점인 'Aggregated Data'를 활용합니다.
    # 각 행은 Match-Team 단위이므로, 여기에 라인별 피처만 추출합니다.
    df_clustering = df[['Match_ID', 'Team_ID'] + feature_cols].copy()

    # NaN 처리
    numeric_cols = df_clustering.select_dtypes(include=np.number).columns
    median_values = df_clustering[numeric_cols].median()
    df_clustering[numeric_cols] = df_clustering[numeric_cols].fillna(median_values)

    X = df_clustering[feature_cols]

    # 표준화 및 클러스터링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=CLUSTER_N, random_state=42, n_init='auto')
    df_clustering['Cluster'] = kmeans.fit_predict(X_scaled)

    return df_clustering, feature_cols


def plot_cluster_heatmap(df_clustering: pd.DataFrame, feature_cols: list):
    """군집별 특징 히트맵 시각화"""

    # 1. 군집별 평균 계산
    df_cluster_mean = df_clustering.groupby('Cluster')[feature_cols].mean()

    # 2. 전체 평균 계산
    overall_mean = df_clustering[feature_cols].mean()

    # 3. 평균 대비 비율 계산 (비율 > 1.0 이면 평균보다 우수)
    df_ratio = df_cluster_mean.div(overall_mean, axis=1)

    # 4. 히트맵을 위해 비율을 로그 변환하여 색상 대비를 명확하게 함 (선택 사항)
    # log2(Ratio)를 사용하면 1.0이 0이 되므로, 0을 중심으로 색상 변화 확인 용이
    df_log_ratio = np.log2(df_ratio)

    # 5. 시각화 준비 (Heatmap)
    plt.figure(figsize=(18, 8))

    # NaN 컬럼 제거 (일부 피처가 모든 Cluster에서 0이거나 NaN일 경우 대비)
    df_log_ratio = df_log_ratio.dropna(axis=1, how='all')

    # 히트맵 제목을 보기 쉽게 라인과 피처로 분리
    clean_cols = [c.replace('_', ' ') for c in df_log_ratio.columns]

    # Vmin/Vmax를 대칭적으로 설정하여 0 (평균)을 중심으로 색상 구분
    v_max = df_log_ratio.abs().max().max()

    sns.heatmap(
        df_log_ratio,
        annot=True,
        cmap='coolwarm',  # 붉은색/푸른색 계열로 긍정/부정(평균 초과/미만)을 구분
        fmt=".2f",
        linewidths=.5,
        linecolor='black',
        vmin=-v_max,
        vmax=v_max,
        xticklabels=clean_cols
    )

    plt.title("Player Style Cluster Feature Heatmap (Log Ratio vs Overall Mean)", fontsize=16)
    plt.ylabel("Cluster ID")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 6. 저장
    out_dir = os.path.join(VISUALIZATION_PATH, "clustering_heatmap")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "cluster_feature_heatmap.png")
    plt.savefig(path, dpi=200)
    print(f"✔ Saved Cluster Heatmap: {path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_FILE_PATH = os.path.join(base_dir, "data", DATA_FILE_PATH)

    df_clustered, feature_cols = prepare_data_and_cluster()

    if df_clustered is not None:
        plot_cluster_heatmap(df_clustered, feature_cols)