import os
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import json

# 🌟 누적/분당 비율 피처로 업데이트 (scoring.py와 동일하게 유지해야 함)
LANE_FEATURE_MAP = {
    "TOP": {"BASE": ["cs", "xp",
                     "ward_place_accum", "ward_kill_accum",
                     "dpm", "kills_accum", "deaths_accum",
                     "assists_accum", "dmg_taken_per_death", "turret_plates_taken",
                     "turret_takedowns_accum", "solo_kills_accum", "split_push_time",
                     "cspm", "kills_per_min"  # 분당 비율 피처
                     ],
            "Early": ["solo_kills_accum"], "Late": ["split_push_time"],
            "End": ["turret_dpm"]},

    "JUNGLE": {
        "BASE": ["jungle_cs", "xp",
                 "ward_place_accum", "ward_kill_accum",
                 "dpm", "kills_accum", "deaths_accum",
                 "assists_accum", "obj_takes_accum", "gank_ka_accum",
                 "kill_participation"  # 분당 비율 피처
                 ],
        "Early": ["gank_ka_accum"],
        "Late": ["kill_participation"],
        "End": ["obj_takes_accum"]},

    "MID": {"BASE": ["cs", "xp",
                     "ward_place_accum", "ward_kill_accum",
                     "dpm", "kills_accum", "deaths_accum",
                     "assists_accum", "roam_ka_accum", "turret_plates_taken",
                     "cspm", "kill_participation"  # 분당 비율 피처
                     ],
            "Early": ["roam_ka_accum"], "Late": ["kill_participation"],
            "End": ["dpm"]},

    "ADC": {"BASE": ["cs", "xp",
                     "ward_place_accum", "ward_kill_accum",
                     "dpm", "kills_accum", "deaths_accum",
                     "assists_accum", "dmg_taken_per_kill", "dmg_dealt_per_death",
                     "total_time_dead",
                     "cspm"  # 분당 비율 피처
                     ],
            "Early": ["cspm"], "Late": ["total_time_dead"],
            "End": ["team_damage_percent"]},

    "SUPPORT": {"BASE": [
        "ward_place_accum", "ward_kill_accum",
        "assists_accum", "roam_ka_accum",
        "heal_per_min", "cc_per_min", "dpm", "kill_participation"  # 분당 비율 피처
    ],
        "Early": ["roam_ka_accum"],
        "Late": ["ward_place_accum", "ward_kill_accum"],  # Late 시야 강화
        "End": ["heal_per_min", "cc_per_min", "kills_per_min", "dpm"]}
}

# [정의] SUPPORT End Phase의 역할별 핵심 피처 맵 (오직 1개만 입력)
SUPPORT_END_FEATURES = {
    "Enchanter": ["heal_per_min"],
    "Tank": ["cc_per_min"],
    "Assassin": ["kills_per_min"],
    "Damage": ["dpm"]
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_one_model(df, lane, phase, role=None):
    save_name = (
        f"{lane}_{role}_{phase}.cbm"
        if role else f"{lane}_{phase}.cbm"
    )
    save_path = os.path.join(MODEL_DIR, save_name)

    # 🌟🌟🌟 수정: base_features와 phase_features를 기본값으로 초기화 🌟🌟🌟
    base_features = []
    phase_features = []
    allowed_features = set()

    # 1. 모델링에 필요한 피처 목록 동적 생성 (Whitelist)
    if lane == "SUPPORT" and phase == "end":
        role_key = role if role in SUPPORT_END_FEATURES else "Damage"
        allowed_features = set(SUPPORT_END_FEATURES[role_key])
        # SUPPORT End Phase는 BASE features를 사용하지 않으므로 빈 리스트로 설정
        base_features = []
        phase_features = list(allowed_features)  # phase_features에 포함
    else:
        base_features = LANE_FEATURE_MAP.get(lane, {}).get("BASE", [])
        phase_features = LANE_FEATURE_MAP.get(lane, {}).get(phase, [])
        allowed_features = set(base_features + phase_features)

    if not allowed_features:
        print(f"⚠️ 학습 건너김: {save_name} (허용 피처가 정의되지 않았습니다.)")
        return

    # 2. 메타데이터 및 불필요 컬럼 제거 (Whitelist Filtering)
    meta_drop_cols = [
        "match_id", "game_id", "pid", "target_gold", "minute", "phase",
        "total_gold", "current_gold", "duration_min", "end_minute",
        "champion", "support_role"
    ]

    X = df.copy()
    y = df["target_gold"]

    all_cols = set(X.columns)

    final_drop_cols = list((all_cols - allowed_features) | set(meta_drop_cols))
    final_drop_cols = [col for col in final_drop_cols if col in X.columns]

    X = X.drop(columns=final_drop_cols, errors="ignore")

    # 3. 데이터 타입 강제 변환 (Key Fix: CatBoostError 해결)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype('float64')

    # 🌟🌟🌟 핵심 수정: 학습 전 X 데이터프레임의 컬럼 순서를 강제 지정 🌟🌟🌟
    # 순서를 강제할 피처 리스트 (CatBoost 오류 방지)
    # BASE features + Phase features의 순서를 따르도록 재구성

    # 🌟 feature_list 구성: 순서를 유지하면서 allowed_features에 있는 피처만 포함
    feature_list = [f for f in (base_features + phase_features) if f in allowed_features and f in X.columns]

    # 순서 재배열
    X = X[feature_list]

    # CatBoost에 전달할 범주형 피처 목록: 이제 'object' 타입은 없어야 함
    cat_features = [col for col in X.columns if X[col].dtype == "object"]

    if len(X) < 10:
        print(f"⚠️ 데이터 부족으로 학습 건너김: {save_name} (row={len(X)})")
        return

    # 4. 학습
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    if X_train.empty:
        print(f"⚠️ 학습 건너김: {save_name} (X_train이 비어있습니다.)")
        return

    # train_pool과 test_pool 생성 시 X의 순서가 유지됩니다.
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.05, loss_function="RMSE",
        early_stopping_rounds=50, verbose=False
    )

    try:
        model.fit(train_pool, eval_set=test_pool)
        model.save_model(save_path)
        print(f"✔ Saved model: {save_path} (RMSE: {model.get_best_score()['validation']['RMSE']:.2f})")
    except Exception as e:
        print(f"❌ 학습 중 에러 발생 ({save_name}): {e}")


def train_all_models(df_early, df_late, df_end):
    print("\n🚀 [Training Start] 총 24개 모델 학습 시작...")

    phases = {"early": df_early, "late": df_late, "end": df_end}
    lanes = ["TOP", "JUNGLE", "MIDDLE", "ADC", "SUPPORT"]

    for phase_name, df_phase in phases.items():
        if df_phase is None or df_phase.empty: continue

        print(f"\n--- Phase: {phase_name} ---")

        for lane in lanes:
            df_lane = df_phase[df_phase["lane"] == lane]
            if df_lane.empty: continue

            if lane == "SUPPORT":
                roles = ["Enchanter", "Tank", "Assassin", "Damage"]
                for r in roles:
                    df_role = df_lane[df_lane["support_role"] == r]
                    if df_role.empty: continue
                    train_one_model(df_role, lane, phase_name, role=r)
            else:
                # 🌟 수정: MIDDLE은 'MID' 키를 사용하도록 매핑
                if lane == "MIDDLE":
                    lane_key = "MID"
                else:
                    lane_key = lane

                train_one_model(df_lane, lane_key, phase_name)

    print("\n🎉 모든 모델 학습 완료!")