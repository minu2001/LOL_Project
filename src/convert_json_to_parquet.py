import os
import json
import glob
import pandas as pd
from tqdm import tqdm

MATCH_JSON_DIR = "raw/match_data"
TIMELINE_JSON_DIR = "raw/timeline_data"

MATCH_PARQUET_DIR = "parquet/match"
TIMELINE_PARQUET_DIR = "parquet/timeline"

os.makedirs(MATCH_PARQUET_DIR, exist_ok=True)
os.makedirs(TIMELINE_PARQUET_DIR, exist_ok=True)


def json_to_parquet(json_path, parquet_path):
    """JSON → Parquet (메모리 절약형 변환)"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # pandas가 JSON 딕셔너리를 잘 flatten 해줌
    df = pd.json_normalize(data)

    # Parquet 저장 (압축 적용)
    df.to_parquet(parquet_path, index=False, compression="snappy")


def batch_convert(json_dir, parquet_dir, prefix):
    files = sorted(glob.glob(os.path.join(json_dir, f"{prefix}_*.json")))

    print(f"Converting {len(files)} files from JSON → Parquet...")

    for path in tqdm(files):
        base = os.path.basename(path).replace(".json", ".parquet")
        parquet_path = os.path.join(parquet_dir, base)

        # 이미 변환된 파일은 skip
        if os.path.exists(parquet_path):
            continue

        json_to_parquet(path, parquet_path)


if __name__ == "__main__":
    batch_convert(MATCH_JSON_DIR, MATCH_PARQUET_DIR, "match")
    batch_convert(TIMELINE_JSON_DIR, TIMELINE_PARQUET_DIR, "timeline")

    print("JSON → Parquet 변환 완료!")
