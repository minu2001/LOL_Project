# 🎮 LOL Player Contribution Analysis
## 하이브리드 기여도 모델 (Manual Weight + CatBoost Residual)

## 1. 프로젝트 개요

이 프로젝트는 리그 오브 레전드(LoL) 플레이어의 **기여도(Contribution Score)**를  
“한 시점에서만 평가하는 기존 OPScore 모델의 한계”를 넘어서기 위해 설계한  
**시간 기반(Time-Split) + Hybrid Weighting 모델**이다.

특히 1000개 이상의 매치 데이터를 기반으로,  
플레이어의 행동을 **시간 흐름에 따라 다르게 해석할 수 있는 구조**를 만드는 것이 핵심이다.

---

## 2. 이 프로젝트의 가장 큰 차별점 (KEY POINTS)

### ✔ A. 0~15분 / 15분 이후 / 종료 시점 → **3단계 Time-Split 모델**
- LoL에서는 초중후반에 같은 행동이라도 “가치”가 달라진다.
- 그래서 모든 라인을 **Early / Late / End** 세 구간으로 나누고  
  각 구간별로 **별도의 CatBoost 모델**을 학습시켰다.
- 이를 통해 **시간 흐름에 따른 기여도 변화를 시계열처럼 추적**할 수 있다.

### ✔ B. “직접 가중치 + CatBoost 자동 가중치”를 합친 **Hybrid 모델**
- Y값이 Gold라서, “골드와 직관적으로 연결된 피처”들에만 가중치가 몰리는 구조를 방지하기 위해  
  일부 피처는 **수작업 가중치(Manual Weight)**를 설정했다.
- 나머지 피처는 CatBoost의 자동 Feature Importance로 학습된다.
- 그래서 “골드와 직접 상관 없는 중요한 행동(무빙 압박, 시야 잡기, 생존 시간 등)”까지 점수화된다.

즉,
> **직접 중요하다고 판단한 피처를 사람이 먼저 잡아주고,  
> 모델은 그 외의 패턴을 자동으로 찾아가는 구조**이다.

### ✔ C. 원래 교수님은 ‘원딜+서폿 묶어서 모델 하나 만들라’고 했음  
근데 우리는 아예 **서폿을 역할군 4개로 분리해서 모델을 따로 만들었다.**
- Support는 다른 라인과 달리 챔피언 역할 편차가 극단적으로 크다.  
  (탱커 / 유틸 / 딜러 / 이니시 / 암살 등)
- 그래서 서폿을 크롤링해 **4개 대분류로 직접 전처리**했고,
  Support 라인도 **역할군별로 각각 early/late/end 모델**을 따로 학습시켰다.

이건 기존 OPScore나 퍼블릭 프로젝트에도 거의 없는 방식이다.

### ✔ D. End Phase에서는 match.json의 “정적 피처”도 모델에 포함 가능
- 종료 시점은 시간 흐름의 중요도가 낮아지고,
- 오히려 **game-end snapshot 데이터** 활용이 의미 있다.
- 타워 피해량, 오브젝트 처치, 최종 KDA 등  
  정적 피처를 End 모델에서만 추가로 활용 가능 → 시계열이 훨씬 정교해짐.

---

## 3. 데이터 구조 및 전체 파이프라인

```
timeline.json → minute-level transform  
 → early/late/end 분할  
 → lane-role classification  
 → manual weights 적용  
 → CatBoost train  
 → prediction merge  
 → final OPScore-like contribution score
```

### (1) 라인 확정
- match.json의 participantId 기반으로 lane 확정

### (2) 시간 단위 분리
- Early:   0 ~ 15분  
- Late:    15분 이후 ~ 거의 끝  
- End:     match.json 기반 정적 스냅샷

### (3) 수작업 가중치 적용
- 내가 중요하다고 판단한 피처만 선별해서  
  early/late/end 구간별로 각각 다른 가중치를 부여
- 피처 예시  
  - TOP → 스플릿푸시 체류시간, 타워딜, 솔킬 기여  
  - MID → 로밍 KA, 라인 압박, DPM  
  - JUNGLE → 카정 체류시간, 갱킹 KA, 오브젝트  
  - ADC → 생존시간, 한타 DPM, CS PM  
  - SUPPORT → 역할군별로 피처 세트 다르게

### (4) CatBoost 자동 가중치
- manual weight로 전체 feature weight를 1.0 중 일부만 사용  
  나머지는 CatBoost에서 자동으로 피처 중요도를 학습

예:
```
manual weight = 0.2  
→ 나머지 0.8은 CatBoost automatic learning
```

### (5) 예측값 병합
- early 모델 예측값 → 초반 분  
- late 모델 예측값 → 중후반 분  
- end 모델 예측값 → 게임 끝 스냅샷  
- 이 3개 모델의 예측을 시간축 위에서 이어 붙여서  
  “게임 전체의 시계열 기여도 그래프”를 만든다.

---

### ✔ 3) 서포터 모델을 4개의 역할군으로 분리

대부분은 바텀(ADC+SUP)을 하나로 묶지만,  
서폿 역할군이 너무 다양하다는 점을 문제로 봤다.

그래서 직접 크롤링 후 support_role.py로 네 가지 역할로 분리했다.

- Enchanter  
- Tank  
- Mage(딜서폿)  
- Assassin(픽형)  

이 작업 덕분에 원딜과 서폿을 분리하여 모델링할 수 있었고  
해석력이 크게 향상됐다.

---
# 3. 라인별 Manual Feature 선정 이유  
본 프로젝트에서는 모든 라인을 **Early(0~15분) / Late(15분 이후) / End(종료 시점)**으로 구분하여  
시간대별로 실제 기여가 달라지는 LoL 구조를 반영하도록 설계했다.

Manual Feature는 “챌린저 표본 분석 + 라인 역할 + 시간대별 중요도 변화”를 기준으로  
**직접 가중치를 부여한 피처**이며, 나머지는 CatBoost가 자동으로 가중치를 학습한다.

---

# 🟦 1) 공통 Manual Feature (전 라인 공통)
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| CS Per Minute | Early / Late | 라인전 파워의 기본 지표 |
| XP Per Minute | Early / Late | 성장 속도 반영 |
| DPM | Late / End | 교전 기여 핵심 지표 |
| Kill / Assist | 전 구간 | 싸움 결과가 기여도에 직접 영향 |
| Death (생존) | Late / End | 후반 생존이 캐리력 좌우 |
| Vision Score | Late / End | 전 라인의 전략적 영향력 반영 |

※ End 페이즈에서는 match.json 기반 최종 스냅샷 값 추가 가능.

---

# 🟥 2) TOP  
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| 솔로킬(Solo Kill) | Early | 초반 라인전 실력 반영 |
| 타워 피해량(Turret Damage) | Late / End | 스플릿 영향력 측정 |
| 스플릿 푸시 체류시간 | Late / End | 탑의 핵심 운영 역할 |
| 받은 피해량(Damage Taken) | Late | 탱킹 기여도 반영 |
| DPM | Late / End | 한타 기여도 점수화 |

---

# 🟩 3) MID  
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| 로밍 Kill/Assist | Early / Late | 미드의 맵 영향력 |
| 라인전 압박(CS·DPM 초기값) | Early | 라인전 푸쉬/주도권 |
| 시야 점수(Vision) | Late | 강가·정글 동선 안정화 |
| 한타 DPM | Late / End | 후반 딜링 기여 핵심 |
| 데스(Death) | Late / End | 미드가 죽으면 전 라인 영향 |

---

# 🟫 4) JUNGLE  
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| 갱킹 Kill/Assist | Early | 라인 스노우볼 형성 |
| 오브젝트 관여도 | Early / Late | 정글러 기여도의 절대 핵심 |
| 카정 체류시간 | Early | 상대 정글 압박 지표 |
| Jungle CS | Early / Late | 정글 성장세 판단 |
| 시야 점수 | Late | 오브젝트 컨트롤 안정화 |

---

# 🟦 5) ADC  
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| CS Per Minute | Early | 원딜의 성장 핵심 |
| Death Time | Late / End | 원딜은 죽지않고 딜을 할 때 포텐이 늘어나는 라인 |
| 한타 DPM | Late / End | 후반 캐리력을 수치화 |
| Kill/Assist | Late | 후반 교전 참여도 |
| 포지셔닝 지표(피해량 대비 생존) | Late / End | 캐리구조 유지 여부 판단 |

---

# 🟪 6) SUPPORT — 4개 역할군 세분화  
Support는 역할 편차가 극단적으로 크므로  
단일 모델보다 **Enchanter / Tank / Mage / Assassin** 4개 모델로 분리하였다.

---

## 6-1) Enchanter Support
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| 힐·쉴드량 | Early / Late | 유틸 성능 핵심 |
| 시야 점수 | 전 구간 | 전략·안정성의 기반 |
| 로밍 보조(K/A) | Late | 미드·정글 동선 보조 |

---

## 6-2) Tank Support
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| CC 시간(TimeCCingOthers) | Early / Late | 이니시·변수 생성 |
| 받은 피해량(Tanked) | Late | 탱커 역할 핵심 |
| 로밍 영향력 | Early | 초중반 싸움 설계 |

---

## 6-3) Mage Support (딜서폿)
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| 포킹 DPM | Early / Late | 라인전 우위 확보 |
| 견제 성공도 | Early | 상대 체력 압박 |
| 시야 점수 | Late | 포킹형 조합 안정화 |

---

## 6-4) Assassin Support (픽형)
| 피처 | 시간대 | 선택 이유 |
|------|--------|------------|
| 픽(Pick) 기여도(K/A·이니시) | Early / Late | 교전 설계 중심 |
| 로밍 성공률 | Early | 변수 창출 능력 |
| CC 활용도 | Late | 후반 교전 영향 |

---

# 4. 전체 모델 Pipeline  
본 프로젝트는 다음 순서로 진행된다.

1. timeline / match 데이터 로드  
2. 분 단위(minute-level) 피처 구성  
3. Early / Late / End 3단계 분리  
4. 각 단계별 Manual Weight 적용  
5. CatBoost 모델을 시간대별로 따로 학습  
6. 세 모델 출력값을 시간축 위에 자연스럽게 연결  
7. 하나의 “기여도 타임라인(Contribution Timeline)” 생성  
8. 라인·경기·소환사 단위 분석 수행  

---

# 5. 시각화 결과 및 그래프 해석

---

## 📊 (1) Feature Definition Radar  
**파일:** `feature_definition_radar.png`  

**이 그래프가 말하는 것**  
- 라인별로 어떤 피처를 강조했는지 한눈에 보인다.  
- Manual Weight 선정 근거를 직관적으로 설명할 수 있다.  
- 특히 Support 4역할군 간 차이가 명확히 드러난다.

---

## 📊 (2) CatBoost Feature Importance  
**파일:** `feature_importance/<lane>_<phase>.png`  

**이 그래프가 말하는 것**  
- CatBoost가 실제로 어떤 피처의 중요도를 높게 평가했는지 보여준다.  
- 사람이 준 manual weight가 실제 모델 구조와 충돌하는지/시너지가 나는지 확인 가능.  
- early/late/end 별로 피처 중요도가 완전히 달라진다는 점이 증명됨.

---

## 📊 (3) Feature Distribution Plot  
**파일:** `distribution/<lane>_<feature>_boxplot.png`  

**이 그래프가 말하는 것**  
- 라인별 분포 차이를 확인 가능  
- 특정 피처가 편향되었는지(=왜 manual weight 적용이 필요했는지) 설명할 수 있음  
- 챌린저 표본의 통계적 범위 파악

---

## 📊 (4) Early vs Late Comparison  
**파일:** `phase_comparison/early_vs_late_score.png`  

**이 그래프가 말하는 것**  
- 각 플레이어가 초반형인지 후반형인지 즉시 파악 가능  
- 정글·미드·서폿은 Early 기여가 높고 원딜은 Late 기여가 높은 정상적인 LoL 메타 구조가 드러남  
- 동일 라인 내에서도 스타일 차이가 극명하게 갈림

---

## 📊 (5) Match Curve (EWMA 시계열 기여도)  
**파일:** `match_curve/match_<id>_ratio.png`  

**이 그래프가 말하는 것**  
- 해당 경기에서 각 라인이 시간대별로 얼마나 기여했는지 직관적으로 보여줌  
- 오브젝트·갱킹·한타 타이밍이 기여도 그래프에 정확히 반영됨  
- 게임 전환점(turning point)을 분석하기 좋음

---

## 📊 (6) Match Pair Curve (양 팀 라인 비교)  
**파일:** `match_pair_curve/match_<id>.png`  

**이 그래프가 말하는 것**  
- 같은 시간대에 Blue/Mid vs Red/Mid 기여도 비교 가능  
- 특정 시간 이후 어느 라인이 캐리했는지 명확히 드러남  
- 승패를 설명하는데 매우 강력한 시각화

---

## 📊 (7) PCA Cluster Map  
**파일:** `clustering_pca/pca_cluster_map.png`  

**이 그래프가 말하는 것**  
- 플레이 스타일(안정형, 폭발형, 성장형 등)을 라벨 없이도 자연스럽게 클러스터링  
- 시간대별 기여 패턴이 구조적으로 묶인다는 것을 확인  
- role 구분 없이도 정글/미드처럼 비슷한 패턴끼리 모임

---

## 📊 (8) Summoner Consistency  
**파일:** `consistency/<puuid>.png`  

**이 그래프가 말하는 것**  
- 소환사의 기여도 안정성을 측정  
- 승리 경기일 때 평균이 높아지는지/기복이 심한지 확인 가능  
- 롤스타급 유저는 그래프가 “1.0 근처에서 평탄하게 유지됨”

---

## 📊 (9) WinPrediction Analysis  
**파일:** `win_prediction/win_prediction_analysis.png`  

**이 그래프가 말하는 것**  
- 기여도 평균만으로도 승패 예측 기반이 형성됨  
- OPScore가 단순 KDA보다 훨씬 승패와 상관관계 높음을 보여줌



---

# 🎯 결론

이 프로젝트는

- 3구간 모델(Early/Late/End)  
- 서포터 역할군 분리  
- Manual Weight + CatBoost Residual  
- Challenger median 정규화  
- 분당 기여도 타임라인 생성  

이라는 구조로 설계된  
가장 현실적인 **라인 기여도 분석 모델**이다.

