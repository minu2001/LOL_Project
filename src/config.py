import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw data
MATCH_PATH = os.path.join(BASE_DIR, "raw/match_data")
TIMELINE_PATH = os.path.join(BASE_DIR, "raw/timeline_data")

# Intermediate data
DATA_DIR = os.path.join(BASE_DIR, "data")
MINUTE_FEATURE_CSV = os.path.join(DATA_DIR, "minute_features.csv")
EARLY_PHASE_CSV = os.path.join(DATA_DIR, "phase_early.csv")
LATE_PHASE_CSV = os.path.join(DATA_DIR, "phase_late.csv")
END_PHASE_CSV = os.path.join(DATA_DIR, "phase_end.csv")

# Models
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Results
RESULT_DIR = os.path.join(BASE_DIR, "results")
VIS_DIR = os.path.join(RESULT_DIR, "visualizations")
FI_DIR = os.path.join(RESULT_DIR, "feature_importance")


LANES = ["TOP", "JUNGLE", "MIDDLE", "ADC", "SUPPORT"]


SUPPORT_ROLE_MAP = {
    "Enchanter": [
        "Nami", "Milio", "Karma", "Janna", "Sona", "Lulu", "Soraka", "Renata",
        "Zilean", "Seraphine", "Yuumi", "Morgana"
    ],
    "Tank": [
        "Thresh", "Leona", "Braum", "Nautilus", "Maokai", "Taric", "Rell",
        "Shen", "Alistar", "Poppy", "Galio", "TahmKench"
    ],
    "Assassin": [
        "Pyke", "Shaco", "LeBlanc"
    ],
    "Damage": [
        "Velkoz", "Brand", "Xerath", "Lux", "Senna", "Zyra", "Pantheon",
        "Nidalee", "Hwaryeong", "Mel"
    ]
}

DEFAULT_SUPPORT_ROLE = "Damage"

VISUALIZATION_PATH = "C:/Users/user/PycharmProjects/Last_LOL_Project/visualizations"