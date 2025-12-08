# src/visualize_core/visualize_feature_radar.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
from math import pi

# 🌟 VISUALIZATION_PATH 설정
VISUALIZATION_PATH = r"C:\Users\user\PycharmProjects\Last_LOL_Project\visualizations"

# 🌟 한글 폰트 설정 (winloss_radar.py 참고)
font_candidates = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
]
KOREAN_FONT_NAME = 'Malgun Gothic'
for fp in font_candidates:
    if os.path.exists(fp):
        KOREAN_FONT_NAME = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['font.family'] = KOREAN_FONT_NAME
plt.rcParams['axes.unicode_minus'] = False

# =======================================================
# [1] 데이터 정의 (엑셀 내용 기반 - 라인별 핵심 피처)
# =======================================================
# 각 라인별로 사용되는 주요 고유 피처들을 정의합니다.
# 값은 임의로 중요도를 나타내는 1로 설정하거나, 사용 여부(0/1)로 설정할 수 있습니다.
# 여기서는 '사용됨'을 의미하는 1.0으로 통일합니다.

lane_features = {
    'TOP': {'솔로킬': 1.0, '스플릿 푸쉬': 1.0, '타워 피해량': 1.0, 'CS': 0.8, 'DPM': 0.7},
    'MID': {'로밍 킬/어시': 1.0, '킬관여율': 1.0, 'DPM': 1.0, 'CS': 0.8, '시야 효율': 0.7},
    'JUNGLE': {'갱킹 K+A': 1.0, '오브젝트 처치': 1.0, '킬관여율': 1.0, '시야 점수': 0.8, 'DPM': 0.6},
    'ADC': {'CS 분당': 1.0, '한타 피해량': 1.0, '생존 시간': 0.9, 'DPM': 1.0, 'KDA': 0.8},
}

# 서포터는 역할군별로 다른 피처 세트를 가집니다.
support_roles = {
    '탱커 (Tank)': {'CC 시간': 1.0, '받은 피해량': 0.9, '이니시': 1.0, '시야 점수': 0.8, '어시스트': 0.7},
    '유틸 (Enchanter)': {'힐/쉴드량': 1.0, '시야 점수': 1.0, '어시스트': 0.9, 'CC 시간': 0.6, '생존': 0.8},
    '딜러 (Mage)': {'DPM': 1.0, '킬/어시': 1.0, '시야 점수': 0.7, '포킹 피해량': 0.9, 'CS': 0.4},
}


def create_radar_chart(ax, categories, values, title, color):
    """단일 레이더 차트 생성 함수"""
    N = len(categories)
    angles = np.linspace(0, 2 * pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, weight='bold')
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["0.5", "1.0"], color="grey", size=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, size=14, weight='bold', y=1.1)

    # 그리드 스타일
    ax.yaxis.grid(True, color='grey', linestyle='--', linewidth=0.5)
    ax.xaxis.grid(True, color='grey', linestyle='--', linewidth=0.5)


def plot_all_lane_radars(save=True):
    print("🚀 라인별 피처 정의 레이더 차트 생성 시작...")

    # 2행 3열의 서브플롯 생성 (탑, 미드, 정글 / 원딜, 서폿)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    # 1. TOP, MID, JUNGLE, ADC 차트 생성
    colors = sns.color_palette("husl", len(lane_features))
    for i, (lane, features) in enumerate(lane_features.items()):
        categories = list(features.keys())
        values = list(features.values())
        create_radar_chart(axes[i], categories, values, f"{lane} 핵심 피처", colors[i])

    # 2. SUPPORT 차트 생성 (역할군별로 겹쳐 그리기)
    sup_ax = axes[4]  # 5번째 서브플롯
    sup_colors = sns.color_palette("Set2", len(support_roles))

    # 모든 서포터 역할군의 피처 합집합을 축으로 사용
    all_sup_features = set()
    for role_features in support_roles.values():
        all_sup_features.update(role_features.keys())
    sup_categories = sorted(list(all_sup_features))

    N = len(sup_categories)
    angles = np.linspace(0, 2 * pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for i, (role, features) in enumerate(support_roles.items()):
        # 해당 역할군에 없는 피처는 0으로 채움
        values = [features.get(cat, 0.0) for cat in sup_categories]
        values += values[:1]

        sup_ax.plot(angles, values, color=sup_colors[i], linewidth=2, label=role)
        sup_ax.fill(angles, values, color=sup_colors[i], alpha=0.1)

    sup_ax.set_xticks(angles[:-1])
    sup_ax.set_xticklabels(sup_categories, size=10, weight='bold')
    sup_ax.set_yticks([0.5, 1.0])
    sup_ax.set_ylim(0, 1.1)
    sup_ax.set_title("SUPPORT 역할군별 피처 (비교)", size=14, weight='bold', y=1.1)
    sup_ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 마지막 빈 서브플롯 제거
    fig.delaxes(axes[5])

    plt.tight_layout()

    # 5. 저장
    if save:
        os.makedirs(VISUALIZATION_PATH, exist_ok=True)
        path = os.path.join(VISUALIZATION_PATH, "feature_definition_radar.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f"✔ Saved Feature Radar Plot: {path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    try:
        plot_all_lane_radars(save=True)
        print("🎉 라인별 피처 레이더 차트 생성 완료!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")