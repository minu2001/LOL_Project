# src/visualize_advanced/pca_cluster_map.py

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================
# [1] 데이터 경로 및 설정
# =======================================================
DATA_FILE_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\data\lol_final_excel_dataset - 복사본.csv"
VISUALIZATION_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\visualizations"
CLUSTER_N = 4


def prepare_clustering_data():
    """lol4.py의 로직: 데이터 로드 및 피처 엔지니어링 수행"""
    print("🚀 STEP 4: 군집 분석을 위한 데이터 준비 시작...")

    if not os.path.exists(DATA_FILE_PATH):
        print(f"❌ 데이터 파일 로드 실패: {DATA_FILE_PATH} 경로에 파일이 없습니다.")
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE_PATH)

    # 군집 분석에 사용할 핵심 피처 그룹
    core_features = {
        'TOP': ['TOP_DPM', 'TOP_KDA', 'TOP_TurretDmg_PM', 'TOP_Split_Time'],
        'JUNGLE': ['JUNGLE_DPM', 'JUNGLE_KDA', 'JUNGLE_Obj_Kills', 'JUNGLE_Gank_KA'],
        'MIDDLE': ['MIDDLE_DPM', 'MIDDLE_KDA', 'MIDDLE_Roam_KA', 'MIDDLE_Vision_Eff'],
        'ADC': ['Adc_DPM', 'Adc_KDA', 'Adc_CS_PM', 'Adc_TeamFight_Dmg'],
        'SUPPORT': ['Sup_DPM', 'Sup_KDA', 'Sup_Ward_Score', 'Sup_Heal_PM'],
    }
    all_features = [f for line in core_features for f in core_features[line]]

    df_features = df[['Match_ID', 'Team_ID'] + all_features].copy()

    # 1차 결측치 처리: 숫자형 피처만 중앙값으로 대체
    numeric_cols = df_features.select_dtypes(include=np.number).columns
    median_values = df_features[numeric_cols].median()
    df_features[numeric_cols] = df_features[numeric_cols].fillna(median_values)

    # 긴 형식으로 변환 및 피벗 (Match_ID + Team_ID + Line을 고유 키로)
    df_long = pd.melt(df_features,
                      id_vars=['Match_ID', 'Team_ID'],
                      value_vars=all_features,
                      var_name='Feature_Line',
                      value_name='Value')

    df_long['Line'] = df_long['Feature_Line'].apply(
        lambda x: x.split('_')[0].replace('Middle', 'MID').replace('Adc', 'ADC').replace('Sup', 'SUPPORT'))
    df_long['Feature'] = df_long['Feature_Line'].apply(lambda x: '_'.join(x.split('_')[1:]))

    df_clustering = df_long.pivot_table(
        index=['Match_ID', 'Team_ID', 'Line'],
        columns='Feature',
        values='Value',
        aggfunc='first'
    ).reset_index()

    # 🚨 최종 수정: Pivot 후 발생한 NaN 재처리 (NaN이 없음을 보장)
    final_numeric_cols = df_clustering.select_dtypes(include=np.number).columns
    final_median_values = df_clustering[final_numeric_cols].median()
    df_clustering[final_numeric_cols] = df_clustering[final_numeric_cols].fillna(final_median_values)

    print(f"✔️ 군집 분석 준비 데이터셋 생성 완료. (Rows: {len(df_clustering)})")

    return df_clustering


def perform_clustering_and_visualize(df_clustering, cluster_n=CLUSTER_N):
    """lol6.py의 로직: K-Means, PCA 및 시각화 수행"""
    print("🚀 STEP 6: 라인별 기여도 패턴 군집 분석 및 PCA 시각화 시작...")

    feature_cols = df_clustering.select_dtypes(include=np.number).columns.tolist()

    X = df_clustering[feature_cols]

    # 1. 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. K-Means 군집 실행
    kmeans = KMeans(n_clusters=cluster_n, random_state=42, n_init='auto')
    df_clustering['Cluster'] = kmeans.fit_predict(X_scaled)

    # 3. PCA (주성분 분석) 실행
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clustering['PC1'] = X_pca[:, 0]
    df_clustering['PC2'] = X_pca[:, 1]

    variance_ratio = pca.explained_variance_ratio_

    # 4. 시각화 (PCA 산점도)
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='tab10',
        data=df_clustering,
        legend='full',
        s=50
    )

    plt.title("Cluster Visualization using PCA (2D) - Player Style Map", fontsize=16)
    plt.xlabel(f"Principal Component 1 ({variance_ratio[0] * 100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({variance_ratio[1] * 100:.1f}%)")
    plt.grid(alpha=0.3)
    plt.legend(title='Cluster ID')
    plt.tight_layout()

    # 5. 저장
    out_dir = os.path.join(VISUALIZATION_PATH, "clustering_pca")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pca_cluster_map.png")
    plt.savefig(path, dpi=200)
    print(f"✔ Saved PCA Cluster Plot: {path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        # 1. 데이터 준비 (lol4.py 기능)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        DATA_FILE_PATH = os.path.join(base_dir, "data", "lol_final_excel_dataset - 복사본.csv")

        df_data = prepare_clustering_data()

        # 2. 군집 분석 및 시각화 (lol6.py 기능)
        if not df_data.empty:
            perform_clustering_and_visualize(df_data)
        else:
            print("❌ 데이터 준비 실패로 PCA 군집 분석을 건너뜁니다.")

    except Exception as e:
        print(f"❌ 단독 실행 중 심각한 오류 발생: {e}")