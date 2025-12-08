"""
support_role.py
서포터 챔피언을 4가지 역할군으로 세분화:
- Enchanter
- Tank
- Assassin
- Damage
"""

ENCHANTER = {
    "Nami", "Milio", "Karma", "Janna", "Soraka", "Sona", "Lulu", "Zilean",
    "Seraphine", "Renata", "Yuumi", "Morgana"
}

TANK = {
    "Thresh", "Leona", "Braum", "Nautilus", "Poppy", "Maokai",
    "Taric", "Rell", "Rakan", "Galio", "TahmKench"
}

ASSASSIN = {
    "Pyke", "Shaco", "Leblanc"
}

DAMAGE = {
    "Velkoz", "Brand", "Zyra", "Xerath", "Lux", "Swain",
    "Pantheon", "Nico", "Teemo", "Hwei", "Elise", "Senna"
}

def get_support_role(champion_name: str):
    """
    챔피언 영어 이름을 받아 4가지 역할 중 하나 반환
    """
    if champion_name in ENCHANTER:
        return "Enchanter"
    if champion_name in TANK:
        return "Tank"
    if champion_name in ASSASSIN:
        return "Assassin"
    if champion_name in DAMAGE:
        return "Damage"

    # 매핑되지 않은 경우 (예: off-meta support)
    return "Damage"   # 딜/포킹 기반으로 기본 분류
