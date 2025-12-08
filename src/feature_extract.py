import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from src.load_data import create_single_mapping

# 설정 파일이 없어도 돌아가도록 임시 처리 (SUPPORT_ROLE_MAP 정의)
try:
    from src.config import MINUTE_FEATURE_CSV, SUPPORT_ROLE_MAP
except ImportError:
    MINUTE_FEATURE_CSV = "data/minute_features.csv"
    SUPPORT_ROLE_MAP = {
        "Enchanter": ["Lulu", "Janna", "Karma", "Nami", "Sona", "Yuumi", "Milio", "Soraka", "Renata", "Seraphine",
                      "Taric", "Zilean", "Ivern"],
        "Tank": ["Leona", "Nautilus", "Thresh", "Blitzcrank", "Alistar", "Rell", "Maokai", "Braum", "TahmKench",
                 "Amumu", "Rakan", "Shen", "Galio", "Poppy", "Sejuani", "Zac", "Gragas", "Sett", "Pantheon"],
        "Assassin": ["Pyke", "Shaco", "Twitch", "Evelynn", "Camille", "Sylas", "Leblanc"]
    }

os.makedirs("data", exist_ok=True)

CHAMP_TO_ROLE = {}
for role, champs in SUPPORT_ROLE_MAP.items():
    for c in champs:
        CHAMP_TO_ROLE[c] = c


def get_support_role(champion_name):
    return CHAMP_TO_ROLE.get(champion_name, "Damage")


def safe_divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def extract_minute_features(match_paths, timeline_paths):
    rows = []

    # KDA/Damage 누적값 초기화 (여기에 와드, 로밍, 오브젝트 누적값 추가)
    accumulators = {
        pid: {"kills": 0, "deaths": 0, "assists": 0,
              "dmg_champ": 0, "dmg_taken": 0, "heal": 0, "cc_time": 0,
              # 🌟 수정: 와드, 로밍, 갱킹, 오브젝트 누적값 추가
              "ward_place": 0, "ward_kill": 0, "roam_ka": 0, "gank_ka": 0, "obj_takes": 0
              }
        for pid in range(1, 11)}

    # 🌟 추가: 이전 프레임의 KDA 누적값 저장을 위한 변수
    prev_kda_accum = {pid: {"kills": 0, "deaths": 0, "assists": 0} for pid in range(1, 11)}

    for idx in tqdm(range(len(match_paths)), desc="Processing Matches"):
        game_id = "UNKNOWN_ID"

        try:
            match_df = pd.read_parquet(match_paths[idx])
            timeline_df = pd.read_parquet(timeline_paths[idx])
            match = match_df.iloc[0].to_dict()
            timeline = timeline_df.iloc[0].to_dict()

            game_id = match.get("metadata.matchId")

            mapping = create_single_mapping(match)

            # accumulator 초기화 및 team_kills 누적 추적
            for key in accumulators:
                for stat in accumulators[key]: accumulators[key][stat] = 0
                # 매치 시작 시 prev_kda_accum도 초기화
                for stat in prev_kda_accum[key]: prev_kda_accum[key][stat] = 0

            team_kills_accum = {100: 0, 200: 0}

            # 정적 정보 추출
            pid_to_info = {}
            participants_info = match.get("info.participants")
            if participants_info is None:
                participants_info = match.get("info", {}).get("participants", [])
            if isinstance(participants_info, np.ndarray): participants_info = participants_info.tolist()

            if participants_info is not None:
                for p in participants_info:
                    pid = p.get("participantId")
                    if pid:
                        challenges = p.get("challenges", {})
                        pid_to_info[pid] = {
                            "target_gold": p.get("goldEarned", 0),
                            "championName": p.get("championName", ""),
                            "turret_plates": challenges.get("turretPlatesTaken", 0),
                            "split_push_time": challenges.get("splitPushTime", 0),
                            "total_time_dead": p.get("totalTimeSpentDead", 0),
                            "team_damage_percent": challenges.get("teamDamagePercentage", 0),
                            "turret_takedowns": p.get("turretTakedowns", 0),
                            "solo_kills": challenges.get("soloKills", 0),
                        }

            # 타임라인 프레임 로딩
            frames = timeline.get("info.frames")
            if frames is None: continue
            if isinstance(frames, np.ndarray): frames = frames.tolist()

            # --- 타임라인 루프 ---
            for frame in frames:
                timestamp_ms = frame.get("timestamp", 0)
                minute = timestamp_ms // 60000

                events = frame.get("events")
                if events is None: events = []
                if isinstance(events, np.ndarray): events = events.tolist()

                pframes = frame.get("participantFrames")
                if pframes is None: pframes = {}

                # 1) 이벤트 집계 및 누적값 업데이트
                player_events_minute = {}

                for ev in events:
                    etype = ev.get("type")

                    # 🌟 수정: WARD_PLACED 이벤트 발생 시 accumulators에 누적
                    if etype == "WARD_PLACED":
                        creator = ev.get("creatorId")
                        if creator in accumulators: accumulators[creator]["ward_place"] += 1

                    # 🌟 수정: WARD_KILL 이벤트 발생 시 accumulators에 누적
                    if etype == "WARD_KILL":
                        killer = ev.get("killerId")
                        if killer in accumulators: accumulators[killer]["ward_kill"] += 1

                    if etype == "CHAMPION_KILL":
                        killer = ev.get("killerId")
                        victim = ev.get("victimId")
                        assists = ev.get("assistingParticipantIds")
                        if assists is None: assists = []
                        if isinstance(assists, np.ndarray): assists = assists.tolist()

                        if killer in mapping["pid_to_team"]:
                            killer_team = mapping["pid_to_team"][killer]
                            team_kills_accum[killer_team] += 1

                        # KDA 누적
                        if killer in accumulators: accumulators[killer]["kills"] += 1
                        if victim in accumulators: accumulators[victim]["deaths"] += 1
                        for ast in assists:
                            if ast in accumulators: accumulators[ast]["assists"] += 1

                        # 🌟 수정: Roam K/A는 accumulators에 누적
                        if killer in accumulators:
                            lane = mapping["pid_to_lane"][killer]
                            if lane in ["MIDDLE", "UTILITY", "SUPPORT"]: accumulators[killer]["roam_ka"] += 1
                            if lane == "JUNGLE": accumulators[killer]["gank_ka"] += 1

                        for ast in assists:
                            if ast in accumulators:
                                lane = mapping["pid_to_lane"][ast]
                                if lane in ["MIDDLE", "UTILITY", "SUPPORT"]: accumulators[ast]["roam_ka"] += 1
                                if lane == "JUNGLE": accumulators[ast]["gank_ka"] += 1

                    if etype == "ELITE_MONSTER_KILL":
                        killer = ev.get("killerId")
                        assists = ev.get("assistingParticipantIds")
                        if assists is None: assists = []
                        if isinstance(assists, np.ndarray): assists = assists.tolist()

                        # 🌟 수정: obj_takes는 accumulators에 누적
                        if killer in accumulators: accumulators[killer]["obj_takes"] += 1
                        for ast in assists:
                            if ast in accumulators: accumulators[ast]["obj_takes"] += 1

                # 2) 플레이어 스탯 업데이트 및 행 추가
                for _, pframe in pframes.items():
                    pid = pframe.get("participantId")
                    if pid not in mapping["pid_to_puuid"]: continue

                    # 🌟🌟🌟 핵심 추가: team_id 추출 🌟🌟🌟
                    team_id = mapping["pid_to_team"].get(pid)
                    if team_id is None: continue  # team_id가 없으면 스킵

                    p_static = pid_to_info.get(pid, {})
                    p_accum = accumulators[pid]

                    # 🌟 분 단위 KDA 계산 (이전 누적값과의 차이)
                    kills_minute = p_accum["kills"] - prev_kda_accum[pid]["kills"]
                    deaths_minute = p_accum["deaths"] - prev_kda_accum[pid]["deaths"]
                    assists_minute = p_accum["assists"] - prev_kda_accum[pid]["assists"]

                    # Raw Stats
                    raw_lane = mapping["pid_to_lane"][pid]
                    if raw_lane == "BOTTOM":
                        lane = "ADC"
                    elif raw_lane == "UTILITY":
                        lane = "SUPPORT"
                    else:
                        lane = raw_lane

                    support_role = get_support_role(p_static.get("championName", ""))

                    cs = pframe.get("minionsKilled", 0)
                    jungle_cs = pframe.get("jungleMinionsKilled", 0)
                    xp = pframe.get("xp", 0)
                    level = pframe.get("level", 0)
                    total_gold = pframe.get("totalGold", 0)
                    current_gold = pframe.get("currentGold", 0)
                    cc_time = pframe.get("timeEnemySpentControlled", 0)

                    dmg_stats = pframe.get("damageStats")
                    if dmg_stats is None: dmg_stats = {}

                    # Accumulators update
                    p_accum["dmg_champ"] = dmg_stats.get("totalDamageDealtToChampions", 0)
                    p_accum["dmg_taken"] = dmg_stats.get("totalDamageTaken", 0)
                    p_accum["heal"] = dmg_stats.get("totalHeal", 0)
                    p_accum["cc_time"] = pframe.get("timeEnemySpentControlled", 0)

                    # Derived Metrics
                    minute_safe = max(1, minute)
                    deaths_accum = p_accum["deaths"]
                    kills_accum = p_accum["kills"]
                    team_total_kills = team_kills_accum.get(mapping["pid_to_team"][pid], 1)

                    dmg_taken_per_death = safe_divide(p_accum["dmg_taken"], deaths_accum)
                    dmg_dealt_per_death = safe_divide(p_accum["dmg_champ"], deaths_accum)
                    dmg_taken_per_kill = safe_divide(p_accum["dmg_taken"], kills_accum)

                    cspm = safe_divide(cs, minute_safe)
                    turret_dpm = safe_divide(p_static.get("turret_damage_total", 0), minute_safe)
                    heal_per_min = safe_divide(p_accum["heal"], minute_safe)
                    cc_per_min = safe_divide(p_accum["cc_time"], minute_safe)
                    kills_per_min = safe_divide(p_accum["kills"], minute_safe)
                    kill_participation = safe_divide(p_accum["kills"] + p_accum["assists"], team_total_kills)

                    rows.append({
                        "match_id": game_id, "minute": minute, "pid": pid,
                        "team_id": team_id,  # 👈 team_id 추가!
                        "champion": p_static.get("championName", ""), "lane": lane,
                        "support_role": support_role, "target_gold": p_static.get("target_gold", 0),

                        # [Raw/Base]
                        "cs": cs, "jungle_cs": jungle_cs, "xp": xp, "level": level,
                        # 🌟 수정: 와드 누적값 사용
                        "ward_place_accum": p_accum["ward_place"],
                        "ward_kill_accum": p_accum["ward_kill"],
                        "dpm": safe_divide(p_accum["dmg_champ"], minute_safe),
                        "kills_accum": kills_accum, "deaths_accum": deaths_accum, "assists_accum": p_accum["assists"],

                        # 🌟 분 단위 KDA 피처 (Manual Score용으로 유지)
                        "kills_minute": kills_minute,
                        "deaths_minute": deaths_minute,
                        "assists_minute": assists_minute,

                        # [Derived Metrics]
                        "dmg_taken_per_death": dmg_taken_per_death,
                        "turret_plates_taken": p_static.get("turret_plates", 0),
                        "turret_takedowns_accum": p_static.get("turret_takedowns", 0),
                        "solo_kills_accum": p_static.get("solo_kills", 0),
                        "split_push_time": p_static.get("split_push_time", 0),
                        "dmg_taken_per_kill": dmg_taken_per_kill,
                        "dmg_dealt_per_death": dmg_dealt_per_death,
                        "total_time_dead": p_static.get("total_time_dead", 0),
                        "team_damage_percent": p_static.get("team_damage_percent", 0),
                        "cspm": cspm, "heal_per_min": heal_per_min, "cc_per_min": cc_per_min,
                        "kills_per_min": kills_per_min, "turret_dpm": turret_dpm,
                        # 🌟 수정: 오브젝트/갱킹/로밍 누적값 사용
                        "obj_takes_accum": p_accum["obj_takes"],
                        "gank_ka_accum": p_accum["gank_ka"],
                        "roam_ka_accum": p_accum["roam_ka"],
                        "kill_participation": kill_participation,

                        "duration_min": match.get("info.gameDuration", 0) // 60
                    })

                    # 🌟 다음 분을 위해 현재 누적값 저장
                    prev_kda_accum[pid]["kills"] = p_accum["kills"]
                    prev_kda_accum[pid]["deaths"] = p_accum["deaths"]
                    prev_kda_accum[pid]["assists"] = p_accum["assists"]

        except Exception as e:
            print(f"❌ Error processing match (ID: {game_id}) at index {idx}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(MINUTE_FEATURE_CSV, index=False)
    return df