#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Level 2 Analysis Final: H2 Lexicality (Auto Loop + Z-scores)
--------------------------------------------------------------------------
Hypothesis 2: Lexicality Effects (Word vs Pseudoword)
Features:
1. Automated Looping: Metrics (Peak/Mean) x SLRT Variables x Data Types (Raw/Zscore).
2. Z-score Calculation (UPDATED): Uses Inverse Normal CDF (qnorm equivalent) for Percentiles.
3. Dynamic Session: Reads session from CSV if available, keeping metadata consistent.
4. Structured Output: Subfolders for Raw vs Zscore analysis.
5. Zero-Size Exclusion: Subjects with Total_Size=0 are EXCLUDED from statistics (Active Only).
6. Master Summary (NEW): Generates a single global CSV containing all results.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm  
import warnings
import re
from itertools import product

warnings.filterwarnings("ignore")

# ============================================================
#  USER CONFIGURATION
# ============================================================

# 1. Metrics to iterate through
METRIC_LIST = ["Mean", "Peak"] 

# 2. SLRT Variables to iterate through
SLRT_VAR_LIST = [
    "SLRT_meanWordsPseudo_listB_PR_R3_T5",
    "SLRT_pseudo_listB_PR_R3_T5",
    "SLRT_words_listB_PR_R3_T5",
    "SLRT_meanWordsPseudo_listB_PR_R3_latestTP"
]

# 3. Data Types to iterate through
DATA_TYPE_LIST = ["Raw", "Zscore"]

# 4. CSV Column Mapping
CSV_COL_MAPPING = {
    "Mean": "Mean", 
    "Peak": "Peak"
}

# 5. Path Settings
BIDS_ROOT = Path("/your/path/")
INPUT_DIR = BIDS_ROOT / "Nilearn"

# Base Output Directory
BASE_OUTPUT_DIR = INPUT_DIR / "Level2_H2_Automated_Zscore"

# 6. Input Files
CLUSTER_CSV = INPUT_DIR / "ROI_Cluster_Analysis_2.3" / "Cluster_Summary_Strict.csv"
PHENO_FILE = INPUT_DIR / "selected.xlsx"
SUBJECT_LIST = INPUT_DIR / "Indivi" / "copied_subjects_list.csv"
MOTION_FILE = INPUT_DIR / "qc_motion_spikes" / "motion_spikes_summary_task-vstring_FDgt1.5.csv"

# 7. H2 DEFINITIONS (Lexicality)
TARGET_HYPOTHESIS = "H2"
TARGET_CONTRASTS = ["word_gt_pseudoword", "pseudoword_gt_word"]
TARGET_ROIS = [
    "L_IFG", "L_STG", 
    "L_vOTC", "L_VWFA1", "L_VWFA2", "L_VWFA_Total"
]

# 8. Covariate Column Names
COL_SUB_PHENO = "subject"
COL_GROUP = "groupByReading"
COL_AGE = "Age_T5"
COL_SEX = "Sex_T5"
COL_HAND = "handedness_writing"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def vp_to_sub(vp):
    m = re.search(r"VP_(\d+)", str(vp))
    return f"sub-{5000 + int(m.group(1))}" if m else None

def encode_sex(val):
    if pd.isna(val): return np.nan
    s = str(val).lower().strip()
    if s.startswith('m'): return 1.0 
    if s.startswith('w') or s.startswith('f'): return 0.0
    return np.nan

def encode_hand(val):
    if pd.isna(val): return np.nan
    s = str(val).lower().strip()
    if 'r' in s: return 1.0 
    if 'l' in s: return 0.0 
    return np.nan

def calculate_weighted_metric(df, metric_type, col_mapping):
    if len(df) == 0: return np.nan
    total_size = df["Size"].sum()
    if total_size == 0: return 0
    
    target_col = col_mapping.get(metric_type)
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in CSV.")
        
    weighted_sum = np.sum(df["Size"] * df[target_col].fillna(0))
    return weighted_sum / total_size

def run_regression(df, y_col, x_cols):
    existing_cols = [y_col] + [c for c in x_cols if c in df.columns]
    data = df[existing_cols].dropna()
    
    if data.shape[0] < len(x_cols) + 2: return None, 0
    
    valid_x = []
    for c in x_cols:
        if c in data.columns and data[c].nunique() > 1:
            valid_x.append(c)
            
    if not valid_x: return None, 0
    
    X = sm.add_constant(data[valid_x])
    y = data[y_col]
    try:
        return sm.OLS(y, X).fit(), len(data)
    except:
        return None, 0

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print(f">>> Loading Base Data (H2)...")

if not CLUSTER_CSV.exists():
    raise FileNotFoundError(f"Cluster CSV not found: {CLUSTER_CSV}")
    
df_raw = pd.read_csv(CLUSTER_CSV)

# Filter Data (H2 only)
df_clusters = df_raw[
    (df_raw["Hypothesis"] == TARGET_HYPOTHESIS) & 
    (df_raw["Contrast"].isin(TARGET_CONTRASTS)) & 
    (df_raw["ROI"].isin(TARGET_ROIS))
].copy()

# Load Pheno
pheno = pd.read_excel(PHENO_FILE)
pheno["Subject"] = pheno[COL_SUB_PHENO].apply(vp_to_sub)

# Pre-process SLRT Columns & Calculate Z-scores (Updated to use norm.ppf)
print("  Processing SLRT Columns & Calculating Z-scores (via norm.ppf)...")
for slrt_col in SLRT_VAR_LIST:
    if slrt_col in pheno.columns:
        pheno[slrt_col] = pd.to_numeric(pheno[slrt_col], errors='coerce')
        slrt_prob = np.clip(pheno[slrt_col] / 100.0, 0.001, 0.999)
        pheno[f"{slrt_col}_z"] = norm.ppf(slrt_prob)

pheno["Age"] = pd.to_numeric(pheno[COL_AGE], errors='coerce')
pheno["Sex_bin"] = pheno[COL_SEX].apply(encode_sex)
pheno["Hand_bin"] = pheno[COL_HAND].apply(encode_hand)
pheno["Group"] = pheno[COL_GROUP].astype(str).str.strip()

# Load Motion
if MOTION_FILE.exists():
    mot = pd.read_csv(MOTION_FILE)
    if not mot["Subject"].iloc[0].startswith("sub-"):
        mot["Subject"] = mot["Subject"].apply(lambda x: f"sub-{x}")
    pheno = pheno.merge(mot[["Subject", "spike_ratio_FD"]], on="Subject", how="left")
else:
    pheno["spike_ratio_FD"] = 0

# Merge Subject List & Dynamic Session Handling
df_subs = pd.read_csv(SUBJECT_LIST)
df_subs["Subject"] = df_subs["Subject"].apply(lambda x: x if str(x).startswith("sub-") else f"sub-{x}")

# --- [新增] 动态提取 Session ---
def extract_session(row):
    for col_name in ["Session", "session", "ses", "Ses"]:
        if col_name in row.index and pd.notna(row[col_name]):
            sess = str(row[col_name]).strip()
            return sess if sess.startswith("ses-") else f"ses-{sess}"
    return "ses-unknown"

if len(df_subs) > 0:
    df_subs["Session"] = df_subs.apply(extract_session, axis=1)
# -----------------------------

df_master_base = df_subs.merge(pheno, on="Subject", how="left")
df_master_base = df_master_base[~df_master_base["Group"].isin(['nan', 'NaN', np.nan])]

print(f"  Valid Subjects: {len(df_master_base)}")
print("="*60)

# ============================================================
# 2. START AUTOMATED LOOPS
# ============================================================

# --- [新增] 全局汇总列表 ---
GLOBAL_MASTER_RESULTS = []

# --- LOOP 1: METRICS (Mean vs Peak) ---
for current_metric in METRIC_LIST:
    print(f"\n>>> PROCESSING METRIC: {current_metric}")
    print("-" * 40)
    
    target_csv_col = CSV_COL_MAPPING.get(current_metric)
    if target_csv_col not in df_raw.columns:
        continue

    # A. Aggregate Clusters
    skeleton = pd.DataFrame(
        list(product(df_master_base["Subject"].unique(), TARGET_CONTRASTS, TARGET_ROIS)),
        columns=["Subject", "Contrast", "ROI"]
    )

    df_agg = df_clusters.groupby(["Subject", "Contrast", "ROI"]).apply(
        lambda x: pd.Series({
            "Total_Size": x["Size"].sum(),
            "Weighted_Value": calculate_weighted_metric(x, current_metric, CSV_COL_MAPPING),
            "Num_Clusters": len(x)
        })
    ).reset_index()

    df_final = skeleton.merge(df_agg, on=["Subject", "Contrast", "ROI"], how="left")
    df_final[["Total_Size", "Weighted_Value"]] = df_final[["Total_Size", "Weighted_Value"]].fillna(0)
    
    df_stats_metric = df_final.merge(df_master_base, on="Subject", how="inner")

    # --- LOOP 2: SLRT VARIABLES ---
    for current_slrt_base in SLRT_VAR_LIST:
        if current_slrt_base not in df_stats_metric.columns:
            continue
            
        # --- LOOP 3: DATA TYPE (Raw vs Zscore) ---
        for data_type in DATA_TYPE_LIST:
            
            analysis_col = current_slrt_base if data_type == "Raw" else f"{current_slrt_base}_z"
            
            if analysis_col not in df_stats_metric.columns:
                continue

            print(f"   -> Analyzing: {current_slrt_base} [{data_type}]")
            
            current_output_dir = BASE_OUTPUT_DIR / current_metric / current_slrt_base / data_type
            current_output_dir.mkdir(parents=True, exist_ok=True)
            
            local_results = []
            
            # B. Run Regressions
            for contrast in TARGET_CONTRASTS:
                con_dir = current_output_dir / contrast
                con_dir.mkdir(exist_ok=True)
                
                for roi in TARGET_ROIS:
                    data = df_stats_metric[(df_stats_metric["Contrast"] == contrast) & (df_stats_metric["ROI"] == roi)].copy()
                    
                    if data["Group"].nunique() >= 2:
                        ref_grp = "normal" if "normal" in data["Group"].unique() else "Typical"
                        data["Group_Bin"] = (data["Group"] != ref_grp).astype(int) 
                    else:
                        data["Group_Bin"] = np.nan
                    
                    for c in ["Age", "spike_ratio_FD"]:
                        if c in data.columns: data[f"{c}_c"] = data[c] - data[c].mean()
                    
                    covs_final = [c for c in ["Age_c", "Sex_bin", "Hand_bin", "spike_ratio_FD_c"] if c in data.columns]
                    
                    data_active = data[data["Total_Size"] > 0].copy()
                    
                    if len(data_active) < 8: continue

                    # [新增] Helper function to populate both lists
                    def add_result(test_name, metric_name, beta_val, p_val, n_val):
                        res_dict = {
                            "Metric": metric_name, "SLRT_Var": current_slrt_base, "Data_Type": data_type,
                            "Contrast": contrast, "ROI": roi, "Test": test_name,
                            "Beta": beta_val, "P": p_val, "Sig": "*" if p_val < 0.05 else "", "N": n_val
                        }
                        local_results.append(res_dict)
                        GLOBAL_MASTER_RESULTS.append(res_dict)

                    # --- 1. INTENSITY ANALYSIS ---
                    if data_active[analysis_col].notna().sum() > 5:
                        model, N = run_regression(data_active, "Weighted_Value", [analysis_col] + covs_final)
                        if model and analysis_col in model.pvalues:
                            p_val = model.pvalues[analysis_col]
                            add_result("Regress_SLRT", f"Intensity_{current_metric}", model.params[analysis_col], p_val, N)
                            
                            plt.figure(figsize=(5,4))
                            sns.regplot(data=data_active, x=analysis_col, y="Weighted_Value", color="royalblue")
                            plt.title(f"{roi} {current_metric} vs {data_type}\n{contrast}\n(Size>0) p={p_val:.3f}")
                            plt.xlabel(f"SLRT ({data_type})")
                            plt.ylabel(f"Weighted {current_metric}")
                            plt.tight_layout()
                            plt.savefig(con_dir / f"Reg_{current_metric}_{roi}.png")
                            plt.close()

                    # Group Diff (Intensity)
                    if data_active["Group_Bin"].nunique() > 1:
                        model, N = run_regression(data_active, "Weighted_Value", ["Group_Bin"] + covs_final)
                        if model and "Group_Bin" in model.pvalues:
                            p_val = model.pvalues["Group_Bin"]
                            add_result("Group_Diff", f"Intensity_{current_metric}", model.params["Group_Bin"], p_val, N)
                            
                            plt.figure(figsize=(4,4))
                            sns.boxplot(data=data_active, x="Group", y="Weighted_Value", palette="Set2")
                            sns.stripplot(data=data_active, x="Group", y="Weighted_Value", color="k", alpha=0.5)
                            plt.title(f"{roi} {current_metric} Group Diff\n{contrast}\n(Size>0) p={p_val:.3f}")
                            plt.tight_layout()
                            plt.savefig(con_dir / f"GroupDiff_{current_metric}_{roi}.png")
                            plt.close()

                    # --- 2. SIZE ANALYSIS ---
                    if data_active[analysis_col].notna().sum() > 5:
                        model, N = run_regression(data_active, "Total_Size", [analysis_col] + covs_final)
                        if model and analysis_col in model.pvalues:
                            p_val = model.pvalues[analysis_col]
                            add_result("Regress_SLRT", "Total_Size_NoZero", model.params[analysis_col], p_val, N)
                            
                            plt.figure(figsize=(5,4))
                            sns.regplot(data=data_active, x=analysis_col, y="Total_Size", color="forestgreen")
                            plt.title(f"{roi} Size vs {data_type}\n{contrast}\n(Size>0) p={p_val:.3f}")
                            plt.xlabel(f"SLRT ({data_type})")
                            plt.ylabel("Cluster Size (vox)")
                            plt.tight_layout()
                            plt.savefig(con_dir / f"Reg_Size_{roi}.png")
                            plt.close()
                    
                    # Group Diff (Size)
                    if data_active["Group_Bin"].nunique() > 1:
                        model, N = run_regression(data_active, "Total_Size", ["Group_Bin"] + covs_final)
                        if model and "Group_Bin" in model.pvalues:
                            p_val = model.pvalues["Group_Bin"]
                            add_result("Group_Diff", "Total_Size_NoZero", model.params["Group_Bin"], p_val, N)
                            
                            plt.figure(figsize=(4,4))
                            sns.boxplot(data=data_active, x="Group", y="Total_Size", palette="Greens")
                            sns.stripplot(data=data_active, x="Group", y="Total_Size", color="k", alpha=0.5)
                            plt.title(f"{roi} Size Group Diff\n{contrast}\n(Size>0) p={p_val:.3f}")
                            plt.tight_layout()
                            plt.savefig(con_dir / f"GroupDiff_Size_{roi}.png")
                            plt.close()
            
            # Save Local Summary CSV
            if local_results:
                df_local = pd.DataFrame(local_results)
                out_file = current_output_dir / f"H2_Results_{current_metric}_{data_type}.csv"
                df_local.sort_values(["Contrast", "ROI"]).to_csv(out_file, index=False)
                print(f"      [Saved Local] {out_file.name}")

# ============================================================
# 3. SAVE GLOBAL MASTER SUMMARY
# ============================================================
if GLOBAL_MASTER_RESULTS:
    df_global = pd.DataFrame(GLOBAL_MASTER_RESULTS)
    
    # 调整列顺序
    col_order = ["Metric", "SLRT_Var", "Data_Type", "Contrast", "ROI", "Test", "Beta", "P", "Sig", "N"]
    final_cols = [c for c in col_order if c in df_global.columns]
    
    # 排序与去重
    df_global = df_global[final_cols].sort_values(["Metric", "SLRT_Var", "Data_Type", "Contrast", "ROI"])
    df_global = df_global.drop_duplicates()
    
    master_out_file = BASE_OUTPUT_DIR / "H2_Master_Summary_All_Results.csv"
    df_global.to_csv(master_out_file, index=False)
    
    print("\n" + "="*60)
    print(f">>> BOOM! Master Summary File Saved to:\n    {master_out_file}")
    print("="*60)
else:
    print("\n>>> Info: No significant results found across all loops.")

print("\n>>> All Automated Loops Completed.")
