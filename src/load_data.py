import json
import os
import glob
import pandas as pd

MATCH_JSON_DIR = "raw/match_data"
TIMELINE_JSON_DIR = "raw/timeline_data"
MATCH_PARQUET_DIR = "parquet/match"
TIMELINE_PARQUET_DIR = "parquet/timeline"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
#  JSON → Parquet 변환
# ============================================================
def convert_json_to_parquet():
    os.makedirs(MATCH_PARQUET_DIR, exist_ok=True)
    os.makedirs(TIMELINE_PARQUET_DIR, exist_ok=True)

    print("🔄 JSON → Parquet 변환 시작...")
    match_files = sorted(glob.glob(os.path.join(MATCH_JSON_DIR, "match_*.json")))
    timeline_files = sorted(glob.glob(os.path.join(TIMELINE_JSON_DIR, "timeline_*.json")))

    for path in match_files:
        out_path = os.path.join(MATCH_PARQUET_DIR, os.path.basename(path).replace(".json", ".parquet"))
        if not os.path.exists(out_path):
            data = load_json(path)
            df = pd.json_normalize(data)
            df.to_parquet(out_path, index=False, compression="snappy")

    for path in timeline_files:
        out_path = os.path.join(TIMELINE_PARQUET_DIR, os.path.basename(path).replace(".json", ".parquet"))
        if not os.path.exists(out_path):
            data = load_json(path)
            df = pd.json_normalize(data)
            df.to_parquet(out_path, index=False, compression="snappy")

    print("✔ JSON → Parquet 변환 완료!")


# ============================================================
# Parquet 파일 경로만 로딩
# ============================================================
def get_parquet_paths():
    match_files = sorted(glob.glob(os.path.join(MATCH_PARQUET_DIR, "match_*.parquet")))
    timeline_files = sorted(glob.glob(os.path.join(TIMELINE_PARQUET_DIR, "timeline_*.parquet")))

    if len(match_files) != len(timeline_files):
        print(f"⚠️ 경고: Match 파일({len(match_files)})과 Timeline 파일({len(timeline_files)}) 개수가 다릅니다.")

    return match_files, timeline_files


# ============================================================
# 단일 매핑 생성 (Numpy-Safe Version)
# ============================================================
def create_single_mapping(match):
    game_id = match.get("metadata.matchId")

    # Flattened Key 확인
    participants = match.get("info.participants")

    # [수정] numpy array 대응: if not participants 대신 is None 사용
    if participants is None:
        # 혹시 모를 Nested 구조 대응
        info = match.get("info")
        if isinstance(info, dict):
            participants = info.get("participants")

    if participants is None:
        raise ValueError(f"❌ match 구조 이상: info.participants 없음 (Match ID: {game_id})")

    puuid_to_pid = {}
    pid_to_puuid = {}
    pid_to_lane = {}
    pid_to_team = {}
    pid_to_champ = {}

    # participants는 리스트 혹은 numpy array이므로 반복문 가능
    for p in participants:
        pid = p.get("participantId")
        puuid = p.get("puuid")

        if pid is None:
            continue

        puuid_to_pid[puuid] = pid
        pid_to_puuid[pid] = puuid
        pid_to_lane[pid] = p.get("teamPosition", "UNKNOWN")
        pid_to_team[pid] = p.get("teamId", 0)
        pid_to_champ[pid] = p.get("championId", 0)

    return {
        "game_id": game_id,
        "puuid_to_pid": puuid_to_pid,
        "pid_to_puuid": pid_to_puuid,
        "pid_to_lane": pid_to_lane,
        "pid_to_team": pid_to_team,
        "pid_to_champ": pid_to_champ
    }