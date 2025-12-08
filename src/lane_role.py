"""
lane_role.py
Riot teamPosition 값의 오류 보정 + ADC/BOTTOM 통합 + 서폿 명확 구분
"""

# Riot teamPosition 문제점:
# - BOTTOM 을 ADC로 줄 때도 있고 SUPPORT 로 줄 때도 있음
# - JUNGLE 은 teamPosition="NONE" 으로 나오는 경우도 흔함
# - TOP/MID 은 비교적 정확


def infer_lane_from_match(participant):
    """
    Riot 제공 teamPosition 값이 부정확한 경우가 많기 때문에
    실제 유저가 플레이한 라인을 재정의한다.
    """

    team_pos = participant.get("teamPosition", "").upper()
    champ = participant.get("championName", "")
    role = participant.get("individualPosition", "").upper()

    # 1) 확실한 포지션은 그대로 사용
    if team_pos in ["TOP", "MIDDLE", "JUNGLE", "INVALID"]:
        if team_pos == "MIDDLE":
            return "MID"
        if team_pos == "INVALID":
            # INVALID 는 대부분 정글이거나 roaming issue
            if role == "JUNGLE":
                return "JUNGLE"
            return "UNKNOWN"
        return team_pos

    # 2) 바텀 듀오 처리
    #    Riot 은 BOTTOM 과 SUPPORT 두 가지를 섞어서 줌
    if team_pos == "BOTTOM":
        return "ADC"  # 원딜로 통일
    if team_pos == "SUPPORT":
        return "SUPPORT"

    # 3) 예외 케이스 처리
    # - 일부 특수 포지션(예: roaming jungle, offrole)
    if role == "JUNGLE":
        return "JUNGLE"
    if role == "SUPPORT":
        return "SUPPORT"
    if role == "BOTTOM":
        return "ADC"

    # 4) 나머지는 champion 기반 추론도 가능하지만
    #    현재 프로젝트에서는 UNKNOWN 처리 → 후처리에서 제거 또는 무시 가능
    return "UNKNOWN"
