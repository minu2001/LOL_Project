"""
manual_rules.py
라인별, Phase별 수동 가중치 계산 로직 (엑셀 파일 반영)
"""

def manual_score(row):
    lane = row["lane"]
    phase = row["phase"]
    score = 0
    # minute이 0이면 나누기 오류가 나므로 최소 1로 설정
    minute = max(1, row["minute"])

    # ------------------------------------------------------
    # 1. 공통 피처 (Common Features)
    # ------------------------------------------------------
    # 기본 점수: KDA + 성장(CS/XP) + 시야 + DPM

    # 🌟🌟🌟 수정 1: KDA 분 단위 이벤트(kills_minute) 사용을 중단하고
    #                분당 비율 지표(kills_per_min)를 사용하도록 변경하여 안정화
    score += row["kills_per_min"] * 0.5 # 분당 킬 비율에 가중치 부여
    score += row["assists_accum"] / minute * 0.25 # 분당 어시스트 비율
    score -= row["deaths_accum"] / minute * 0.5 # 분당 데스 비율

    # DPM (약 500~800 -> 0.005 곱하면 2.5~4점)
    score += row["dpm"] * 0.015

    # 🌟🌟🌟 수정 2: 와드 누적값(accum)을 분당 비율로 변환하여 사용 🌟🌟🌟
    ward_per_min = (row["ward_place_accum"] + row["ward_kill_accum"]) / minute
    score += ward_per_min * 0.5 # 분당 비율이므로 0.5 가중치는 적절함

    # ------------------------------------------------------
    # 2. 라인별 고유/Phase 피처 (Lane Specific)
    # ------------------------------------------------------

    if lane == "TOP":
        if phase == "early":
            # 솔로킬 (누적값) - 분당 솔로킬로 해석하여 가중치 하향
            score += (row["solo_kills_accum"] / minute) * 0.5
            # 포탑 방패 (누적값) - 누적값이므로 가중치 하향
            score += row["turret_plates_taken"] * 0.05

        elif phase == "late":
            # 스플릿 푸쉬 시간 (0.5 가산 유지)
            if row["split_push_time"] > 0:
                score += 0.5

        elif phase == "end":
            score += row["turret_dpm"] * 0.015

    elif lane == "MID":
        if phase == "early":
            # 로밍 킬/어시 (누적값) - 분당 비율로 해석하여 가중치 하향
            score += (row["roam_ka_accum"] / minute) * 0.5
            # 포탑 방패 (누적값) - 누적값이므로 가중치 하향
            score += row["turret_plates_taken"] * 0.05

        elif phase == "late":
            # 킬 관여율 (KP) - 비율이므로 1.0 유지
            score += row["kill_participation"] * 1.0

        elif phase == "end":
            score += row["dpm"] * 0.002

    elif lane == "JUNGLE":
        if phase == "early":
            # 갱킹 킬+어시 (누적값) - 분당 비율로 해석하여 가중치 하향
            score += (row["gank_ka_accum"] / minute) * 0.5

        elif phase == "late":
            # 킬 관여율 (KP) - 비율이므로 1.0 유지
            score += row["kill_participation"] * 1.0

        elif phase == "end":
            # 오브젝트 처치 (누적값) - 분당 비율로 해석하여 가중치 대폭 하향
            score += (row["obj_takes_accum"] / minute) * 1.0

    elif lane == "ADC":
        if phase == "early":
            # 분당 CS (cspm) - 비율이므로 0.5 유지
            score += row["cspm"] * 0.5

        elif phase == "late":
            pass

        elif phase == "end":
            # 한타 피해량 (Teamfight Damage) - 비율이므로 0.1 유지
            score += (row["team_damage_percent"] * 100) * 0.1

    elif lane == "SUPPORT":
        role = row["support_role"]

        if phase == "early":
            # 로밍 킬 + 어시 (누적값) - 분당 비율로 해석하여 가중치 하향
            score += (row["roam_ka_accum"] / minute) * 0.5

        elif phase == "late":
            # 시야 장악 (와드 분당 비율) - 이미 공통에서 계산했으므로 추가 가산 없이 공통의 ward_per_min 사용
            pass

        elif phase == "end":
            # 역할군별 핵심 지표 (모두 분당 비율이므로 유지)
            if role == "Enchanter":
                score += row["heal_per_min"] * 0.01
            elif role == "Tank":
                score += row["cc_per_min"] * 2.0
            elif role == "Assassin":
                score += row["kills_per_min"] * 2.0
            elif role == "Damage":
                score += row["dpm"] * 0.015

    return score