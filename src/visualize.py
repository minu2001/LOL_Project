import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

# 시각화 저장 경로 설정
VIS_DIR = "visuals"
os.makedirs(VIS_DIR, exist_ok=True)

def visualize_feature_importance(model_dir="models"):
    """
    저장된 CatBoost 모델들의 피처 중요도를 시각화하여 저장합니다.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("CatBoost not installed. Skipping feature importance visualization.")
        return

    print("\n📊 Generating Feature Importance Plots...")
    
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found.")
        return

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".cbm")]
    
    if not model_files:
        print("No model files found.")
        return

    for model_file in model_files:
        try:
            model_path = os.path.join(model_dir, model_file)
            model = CatBoostRegressor()
            model.load_model(model_path)
            
            # 피처 중요도 추출
            feature_importance = model.get_feature_importance()
            feature_names = model.feature_names_
            
            # DataFrame 생성 및 상위 10개 추출
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
            fi_df = fi_df.sort_values(by='importance', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x="importance", y="feature", data=fi_df, palette="viridis", hue="feature", legend=False)
            plt.title(f"Feature Importance - {model_file}")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            
            save_path = os.path.join(VIS_DIR, f"importance_{model_file}.png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"✔ Saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to plot {model_file}: {e}")

def visualize_opscore_distribution(df):
    """
    라인별 OPScore 분포를 히스토그램으로 시각화합니다.
    """
    print("\n📊 Generating OPScore Distribution Plot...")

    if df is None or df.empty:
        print("⚠️ No data to visualize.")
        return

    # [핵심 수정] scoring.py에서 만든 컬럼명은 'op_score'입니다.
    target_col = "op_score" 
    
    if target_col not in df.columns:
        print(f"❌ Error: '{target_col}' column not found. Available columns: {list(df.columns)}")
        # 혹시 모를 이름 불일치 대비 (opscore가 있다면 그걸 사용)
        if "opscore" in df.columns:
            target_col = "opscore"
            print(f"   -> Found 'opscore' instead. Using it.")
        else:
            return

    plt.figure(figsize=(12, 7))
    
    # 라인별로 반복하여 히스토그램 그리기
    lanes = df["lane"].unique()
    for lane in lanes:
        subset = df[df["lane"] == lane]
        data = subset[target_col].dropna()
        
        if len(data) > 0:
            sns.histplot(data, kde=True, label=lane, element="step", alpha=0.5)

    plt.title("OPScore Distribution by Lane")
    plt.xlabel("OPScore (Contribution Score)")
    plt.ylabel("Frequency")
    plt.legend(title="Lane")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(VIS_DIR, "opscore_distribution.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"✔ Saved distribution plot: {save_path}")