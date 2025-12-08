import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from src.manual_rules import manual_score
import json

MODEL_DIR = "models"

# [정의] SUPPORT End Phase의 역할별 핵심 피처 맵 (model_training.py와 동일하게 유지)
SUPPORT_END_FEATURES = {
    "Enchanter": ["heal_per_min"],
    "Tank": ["cc_per_min"],
    "Assassin": ["kills_per_min"],
    "Damage": ["dpm"]
}

# 🌟 누적/분당 비율 피처로 업데이트 (model_training.py와 동일)
LANE_FEATURE_MAP = {
    "TOP": {"BASE": ["cs", "xp",
                     "ward_place_accum", "ward_kill_accum",
                     "dpm", "kills_accum", "deaths_accum",
                     "assists_accum", "dmg_taken_per_death", "turret_plates_taken",
                     "turret_takedowns_accum", "solo_kills_accum", "split_push_time",
                     "cspm", "kills_per_min"
                     ],
            "Early": ["solo_kills_accum"], "Late": ["split_push_time"],
            "End": ["turret_dpm"]},

    "JUNGLE": {
        "BASE": ["jungle_cs", "xp",
                 "ward_place_accum", "ward_kill_accum",
                 "dpm", "kills_accum", "deaths_accum",
                 "assists_accum", "obj_takes_accum", "gank_ka_accum",
                 "kill_participation"
                 ],
        "Early": ["gank_ka_accum"],
        "Late": ["kill_participation"],
        "End": ["obj_takes_accum"]},

    "MID": {"BASE": ["cs", "xp",
                     "ward_place_accum", "ward_kill_accum",
                     "dpm", "kills_accum", "deaths_accum",
                     "assists_accum", "roam_ka_accum", "turret_plates_taken",
                     "cspm", "kill_participation"
                     ],
            "Early": ["roam_ka_accum"], "Late": ["kill_participation"],
            "End": ["dpm"]},

    "ADC": {"BASE": ["cs", "xp",
                     "ward_place_accum", "ward_kill_accum",
                     "dpm", "kills_accum", "deaths_accum",
                     "assists_accum", "dmg_taken_per_kill", "dmg_dealt_per_death",
                     "total_time_dead",
                     "cspm"
                     ],
            "Early": ["cspm"], "Late": ["total_time_dead"],
            "End": ["team_damage_percent"]},

    "SUPPORT": {"BASE": [
        "ward_place_accum", "ward_kill_accum",
        "assists_accum", "roam_ka_accum",
        "heal_per_min", "cc_per_min", "dpm", "kill_participation"
    ],
        "Early": ["roam_ka_accum"],
        "Late": ["ward_place_accum", "ward_kill_accum"],
        "End": ["heal_per_min", "cc_per_min", "kills_per_min", "dpm"]}
}


def add_phase_column(df):
    conditions = [
        (df["minute"] < 15),
        (df["minute"] >= df["duration_min"])
    ]
    choices = ["early", "end"]
    df["phase"] = np.select(conditions, choices, default="late")
    return df


def compute_opscore(df):
    if "phase" not in df.columns:
        df = add_phase_column(df)

    df["model_score"] = 0.0
    df["manual_score"] = 0.0
    df["op_score"] = 0.0

    if "support_role" not in df.columns:
        df["support_role"] = "None"

    # 'MIDDLE' 학습 모델은 'MID' 키로 저장되었으므로 그룹핑 시 'lane'을 수정해야 함
    df['lane_model_key'] = df['lane'].replace('MIDDLE', 'MID')

    groups = df.groupby(["lane_model_key", "phase", "support_role"])

    # 학습 때 제외했던 메타 컬럼들 (model_training.py와 동일해야 함)
    meta_drop_cols = [
        "match_id", "game_id", "pid",
        "target_gold", "minute", "phase",
        "total_gold", "current_gold",
        "duration_min", "end_minute",
        "champion", "support_role", "lane_model_key"
    ]

    print("🚀 OPScore 계산 시작 (Batch Prediction)...")

    for (lane, phase, role), group_indices in groups.groups.items():
        if len(group_indices) == 0:
            continue

        subset = df.loc[group_indices].copy()

        if lane == "SUPPORT":
            model_name = f"{lane}_{role}_{phase}.cbm"
        else:
            # 🌟 수정: MIDDLE 포지션도 MID 키로 모델명을 찾음
            model_name = f"{lane}_{phase}.cbm"

        model_path = os.path.join(MODEL_DIR, model_name)

        if os.path.exists(model_path):
            try:
                model = CatBoostRegressor()
                model.load_model(model_path)

                # 1. 허용 피처 목록 동적 생성 (Feature Selection)
                if lane == "SUPPORT" and phase == "end":
                    # SUPPORT End Phase는 핵심 피처 1개만 선택
                    role_key = role if role in SUPPORT_END_FEATURES else "Damage"
                    allowed_features = set(SUPPORT_END_FEATURES[role_key])
                    # End Phase는 BASE features가 없으므로 feature_list를 직접 구성
                    base_features = []
                else:
                    # 그 외는 Base + Phase Features 모두 사용
                    base_features = LANE_FEATURE_MAP.get(lane, {}).get("BASE", [])
                    phase_features = LANE_FEATURE_MAP.get(lane, {}).get(phase, [])
                    allowed_features = set(base_features + phase_features)

                # 2. 드롭할 컬럼 목록 최종 결정 (학습 때와 동일하게)
                all_subset_cols = set(subset.columns)
                cols_to_drop = list((all_subset_cols - allowed_features) | set(meta_drop_cols))

                X_subset = subset.drop(columns=cols_to_drop, errors="ignore")

                # 🌟🌟🌟 핵심 수정: 예측 전 X_subset의 컬럼 순서를 강제 지정 🌟🌟🌟
                # 학습 시와 동일하게 feature_list를 BASE와 Phase features를 기반으로 순서 지정
                feature_list = [f for f in (base_features + phase_features) if
                                f in allowed_features and f in X_subset.columns]

                # 순서 재배열
                X_subset = X_subset[feature_list]

                # 3. 예측
                preds = model.predict(X_subset)
                df.loc[group_indices, "model_score"] = preds

            except Exception as e:
                print(f"⚠️ 모델 예측 실패 ({model_name}): {e}")
        else:
            # 모델 파일이 없으면 0.0으로 유지되고 manual_score만 사용됨
            pass

    print("   -> Calculating manual scores...")
    df["manual_score"] = df.apply(manual_score, axis=1)

    print("   -> Finalizing OPScore...")
    end_mask = (df["phase"] == "end")
    not_end_mask = ~end_mask

    # ➡️ End Phase 가중치 (0.2/0.8로 유지)
    df.loc[end_mask, "op_score"] = (
            0.2 * df.loc[end_mask, "manual_score"] +
            0.8 * df.loc[end_mask, "model_score"]
    )

    # ➡️ Early/Late Phase 가중치 (0.1/0.9로 유지)
    df.loc[not_end_mask, "op_score"] = (
            0.1 * df.loc[not_end_mask, "manual_score"] +
            0.9 * df.loc[not_end_mask, "model_score"]
    )

    # 🌟🌟🌟 핵심 추가: OPScore를 final_score_norm으로 정규화하여 저장 🌟🌟🌟
    print("   -> Normalizing final score...")

    # OPScore를 최종적으로 0과 1 사이로 Min/Max 정규화
    op_min = df["op_score"].min()
    op_max = df["op_score"].max()

    if op_max > op_min:
        df["final_score_norm"] = (df["op_score"] - op_min) / (op_max - op_min)
    else:
        # 모든 값이 같거나 0인 경우 0으로 처리 (나누기 0 방지)
        df["final_score_norm"] = 0.0

    return df