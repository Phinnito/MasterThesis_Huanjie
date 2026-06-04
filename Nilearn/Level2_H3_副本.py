#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Level 2 Analysis H3: ROI & Whole-Brain Laterality Comparison
------------------------------------------------------------------
Upgrades:
1. ROI Update: Added 'vOTC' (bilateral) to mask definitions.
2. Statistics: Added One-Sample T-Test (vs 0) to check distinct lateralization.
3. Visualization: Added histograms for population-level LI distribution.
4. Data Transform: Added Z-score standardization from Percentile (SLRT -> SLRT_z).
5. Dynamic Session: Automatically detects session ID from CSV or folder structure.

Logic:
1. Loads Level 1 Z-maps.
2. Calculates Weighted LI (Sum Positive Z).
3. Step 1: One-Sample T-Test (Is the region lateralized?).
4. Step 2: Regression (Does LI correlate with Behavior? Percentile & Z-score).
5. Step 3: Group Difference (Do groups differ in LI?).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_1samp, norm
import warnings
import re
from nilearn.image import load_img, math_img, new_img_like
from nilearn.masking import apply_mask

warnings.filterwarnings("ignore")

# ============================================================
#  USER CONFIGURATION
# ============================================================

# 1. Path Settings
BIDS_ROOT = Path("/your/path/")
OUTPUT_ROOT = BIDS_ROOT / "Nilearn" / "Level2"/ "H3_vOTC_OneSample"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

GLM_DIR = BIDS_ROOT / "Nilearn" / "nilearn_glm_level1_new"
MASK_DIR = BIDS_ROOT / "Nilearn" / "ROI_Mask"

PHENO_FILE = BIDS_ROOT / "Nilearn" / "selected.xlsx"
SUBJECT_LIST = BIDS_ROOT / "Nilearn" / "Indivi" / "copied_subjects_list.csv"
MOTION_FILE = BIDS_ROOT / "Nilearn" / "qc_motion_spikes" / "motion_spikes_summary_task-vstring_FDgt1.5.csv"

# 2. Base ROI Definitions
BASE_ROIS = {
    "VWFA1": {
        "L": MASK_DIR / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_L_VWFA1.nii.gz",
        "R": MASK_DIR / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_R_VWFA1.nii.gz"
    },
    "VWFA2": {
        "L": MASK_DIR / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_L_VWFA2.nii.gz",
        "R": MASK_DIR / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_R_VWFA2.nii.gz"
    },
    "vOTC": {
        "L": MASK_DIR / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_L_vOTC.nii.gz",
        "R": MASK_DIR / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_R_vOTC.nii.gz"
    },
    "IFG": {
        "L": MASK_DIR / "HO_IFG_L_thr25_MNIPed.nii.gz",
        "R": MASK_DIR / "HO_IFG_R_thr25_MNIPed.nii.gz"
    },
    "STG": {
        "L": MASK_DIR / "HO_STG_L_thr25_MNIPed.nii.gz",
        "R": MASK_DIR / "HO_STG_R_thr25_MNIPed.nii.gz"
    }
}

# 3. Combined ROI Definitions (Auto-Merge)
COMBINED_ROIS = {
    "VWFA_Total": ["VWFA1", "VWFA2"]
}

# 4. Include WholeBrain Analysis?
INCLUDE_WHOLEBRAIN = True  

# 5. Analysis Parameters
TARGET_CONTRASTS = [
    "word_gt_falsefont", 
    "pseudoword_gt_falsefont", 
    "word_gt_pseudoword"
]

LI_THRESHOLD = 0 

# 6. Phenotype Column Names
COL_SUB_PHENO = "subject"
COL_SCORE = "SLRT_meanWordsPseudo_listB_PR_R3_latestTP"
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

_WHOLEBRAIN_MASKS_CACHE = None
def generate_hemi_masks(ref_img_path):
    ref_img = load_img(str(ref_img_path))
    data = ref_img.get_fdata()
    affine = ref_img.affine
    indices = np.indices(data.shape)
    x_coords = (indices[0] * affine[0, 0] + indices[1] * affine[0, 1] + 
                indices[2] * affine[0, 2] + affine[0, 3])
    l_mask_data = (x_coords < -2).astype(int)
    r_mask_data = (x_coords > 2).astype(int)
    return new_img_like(ref_img, l_mask_data), new_img_like(ref_img, r_mask_data)

def get_roi_masks(roi_name, base_rois, combined_rois, ref_z_map=None):
    global _WHOLEBRAIN_MASKS_CACHE
    if roi_name == "WholeBrain":
        if _WHOLEBRAIN_MASKS_CACHE is None:
            if ref_z_map is None: raise ValueError("Need ref_z_map for WholeBrain.")
            print("    [Info] Generating WholeBrain masks...")
            _WHOLEBRAIN_MASKS_CACHE = generate_hemi_masks(ref_z_map)
        return _WHOLEBRAIN_MASKS_CACHE

    if roi_name in base_rois:
        return load_img(str(base_rois[roi_name]["L"])), load_img(str(base_rois[roi_name]["R"]))
    
    if roi_name in combined_rois:
        sub_names = combined_rois[roi_name]
        l_imgs = [str(base_rois[n]["L"]) for n in sub_names]
        r_imgs = [str(base_rois[n]["R"]) for n in sub_names]
        formula = " + ".join([f"img{i+1}" for i in range(len(l_imgs))])
        l_merged = math_img(formula, **{f"img{i+1}": load_img(i) for i, img in enumerate(l_imgs)})
        r_merged = math_img(formula, **{f"img{i+1}": load_img(i) for i, img in enumerate(r_imgs)})
        return l_merged, r_merged
    raise ValueError(f"Unknown ROI: {roi_name}")

def calculate_li(z_map_path, l_mask, r_mask, threshold=0):
    try:
        z_img = load_img(str(z_map_path))
        vals_l = apply_mask(z_img, l_mask)
        vals_r = apply_mask(z_img, r_mask)
        
        sum_l = np.sum(vals_l[vals_l > threshold])
        sum_r = np.sum(vals_r[vals_r > threshold])
        
        if (sum_l + sum_r) == 0: return np.nan, 0, 0
            
        li = (sum_l - sum_r) / (sum_l + sum_r)
        return li, sum_l, sum_r
    except Exception as e:
        return np.nan, 0, 0

def run_regression(df, y_col, x_cols):
    existing_cols = [y_col] + [c for c in x_cols if c in df.columns]
    data = df[existing_cols].dropna()
    if data.shape[0] < len(x_cols) + 2: return None, 0
    valid_x = [c for c in x_cols if c in data.columns and data[c].nunique() > 1]
    if not valid_x: return None, 0
    X = sm.add_constant(data[valid_x])
    y = data[y_col]
    try:
        return sm.OLS(y, X).fit(), len(data)
    except:
        return None, 0

# ============================================================
# 1. LOAD DATA
# ============================================================
print(">>> Loading Data...")

pheno = pd.read_excel(PHENO_FILE)
pheno["Subject"] = pheno[COL_SUB_PHENO].apply(vp_to_sub)
pheno["SLRT"] = pd.to_numeric(pheno[COL_SCORE], errors='coerce')

# Convert Percentile to Z-score
slrt_prob = np.clip(pheno["SLRT"] / 100.0, 0.001, 0.999)
pheno["SLRT_z"] = norm.ppf(slrt_prob)

pheno["Age"] = pd.to_numeric(pheno[COL_AGE], errors='coerce')
pheno["Sex_bin"] = pheno[COL_SEX].apply(encode_sex)
pheno["Hand_bin"] = pheno[COL_HAND].apply(encode_hand)
pheno["Group"] = pheno[COL_GROUP].astype(str).str.strip()

if MOTION_FILE.exists():
    mot = pd.read_csv(MOTION_FILE)
    if not mot["Subject"].iloc[0].startswith("sub-"):
        mot["Subject"] = mot["Subject"].apply(lambda x: f"sub-{x}")
    pheno = pheno.merge(mot[["Subject", "spike_ratio_FD"]], on="Subject", how="left")
else:
    pheno["spike_ratio_FD"] = 0

df_subs = pd.read_csv(SUBJECT_LIST)
df_subs["Subject"] = df_subs["Subject"].apply(lambda x: x if str(x).startswith("sub-") else f"sub-{x}")
df_master = df_subs.merge(pheno, on="Subject", how="left")
df_master = df_master[~df_master["Group"].isin(['nan', 'NaN', np.nan])]
print(f"  Valid Subjects: {len(df_master)}")

# ============================================================
# 2. CALCULATE LI
# ============================================================
print(f">>> Calculating LI (Threshold Z > {LI_THRESHOLD})...")

roi_list = list(BASE_ROIS.keys()) + list(COMBINED_ROIS.keys())
if INCLUDE_WHOLEBRAIN: roi_list.append("WholeBrain")
print(f"  Target ROIs: {roi_list}")

li_data = []
for idx, row in df_master.iterrows():
    sub = row["Subject"]
    sess = None
    
    # --- [新增] 动态读取 Session ---
    # 策略 1: 尝试从 CSV 数据中读取可能存在的 session 列
    for col_name in ["Session", "session", "ses", "Ses"]:
        if col_name in row.index and pd.notna(row[col_name]):
            sess = str(row[col_name]).strip()
            # 确保格式是 ses-X
            if not sess.startswith("ses-"):
                sess = f"ses-{sess}"
            break
            
    # 策略 2: 如果 CSV 里没有，直接去受试者文件夹里找对应的 ses- 文件夹
    if not sess:
        sub_dir = GLM_DIR / sub
        if sub_dir.exists():
            ses_dirs = list(sub_dir.glob("ses-*"))
            if ses_dirs:
                sess = ses_dirs[0].name # 取找到的第一个 ses- 文件夹
    
    # 终极 Fallback
    if not sess:
        print(f"    [Warning] Could not detect session for {sub}. Assuming ses-1.")
        sess = "ses-1"
    # -----------------------------------

    sub_glm_dir = GLM_DIR / sub / sess / "H1"
    first_z_map = None
    
    for contrast in TARGET_CONTRASTS:
        z_fname = f"{sub}_{sess}_task-vstring_contrast-{contrast}_zmap.nii.gz"
        z_path = sub_glm_dir / z_fname
        if not z_path.exists(): continue
        if first_z_map is None: first_z_map = z_path
            
        for roi_name in roi_list:
            try:
                l_mask, r_mask = get_roi_masks(roi_name, BASE_ROIS, COMBINED_ROIS, ref_z_map=first_z_map)
                li, l_sum, r_sum = calculate_li(z_path, l_mask, r_mask, threshold=LI_THRESHOLD)
                li_data.append({
                    "Subject": sub, "Session": sess, "Contrast": contrast, "ROI": roi_name,
                    "LI": li, "Sum_L": l_sum, "Sum_R": r_sum
                })
            except Exception: pass

df_li = pd.DataFrame(li_data)
# 融合时剔除之前可能从文件里带来的老 Session 列，防止合并冲突
df_stats = df_li.merge(df_master.drop(columns=["Session", "session", "ses"], errors="ignore"), on="Subject", how="inner")
df_stats.to_csv(OUTPUT_ROOT / "H3_All_LI_Data.csv", index=False)

# ============================================================
# 3. STATISTICAL ANALYSIS (One-Sample T -> Regression -> Group)
# ============================================================
print(">>> Running Stats (One-Sample T -> Regressions)...")

results = []

for contrast in TARGET_CONTRASTS:
    con_dir = OUTPUT_ROOT / contrast
    con_dir.mkdir(exist_ok=True)
    
    for roi_name in roi_list:
        data = df_stats[(df_stats["Contrast"] == contrast) & (df_stats["ROI"] == roi_name)].copy()
        data_valid = data.dropna(subset=["LI"]).copy()
        
        if len(data_valid) < 8: continue

        # --- STEP 1: ONE-SAMPLE T-TEST ---
        t_stat, p_val_1samp = ttest_1samp(data_valid["LI"], 0)
        mean_li = data_valid["LI"].mean()
        
        results.append({
            "Contrast": contrast, "ROI": roi_name, "Test": "OneSample_MeanLI",
            "Beta": mean_li, "P": p_val_1samp, "N": len(data_valid),
            "Sig": "*" if p_val_1samp < 0.05 else ""
        })
        
        plt.figure(figsize=(5, 4))
        sns.histplot(data=data_valid, x="LI", kde=True, color="teal", bins=15)
        plt.axvline(0, color='k', linestyle='--', linewidth=1, label="Zero")
        plt.axvline(mean_li, color='r', linestyle='-', linewidth=1.5, label=f"Mean={mean_li:.2f}")
        plt.title(f"{roi_name} LI Distribution\nMean={mean_li:.2f}, p={p_val_1samp:.3f}")
        plt.xlabel("LI (-1 Right | +1 Left)")
        plt.xlim(-1.1, 1.1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(con_dir / f"OneSample_Dist_{roi_name}.png")
        plt.close()

        # --- PREPARE FOR REGRESSION ---
        if data["Group"].nunique() >= 2:
            unique_groups = data["Group"].unique()
            if "normal" in unique_groups: ref_grp = "normal"
            elif "Typical" in unique_groups: ref_grp = "Typical"
            else: ref_grp = sorted(unique_groups)[0]
            data_valid["Group_Bin"] = (data_valid["Group"] != ref_grp).astype(int)
        else:
            data_valid["Group_Bin"] = np.nan
        
        for c in ["Age", "spike_ratio_FD"]:
            if c in data_valid.columns: data_valid[f"{c}_c"] = data_valid[c] - data_valid[c].mean()
            
        covs_final = []
        if "Age_c" in data_valid.columns: covs_final.append("Age_c")
        if "Sex_bin" in data_valid.columns: covs_final.append("Sex_bin")
        if "Hand_bin" in data_valid.columns: covs_final.append("Hand_bin")
        if "spike_ratio_FD_c" in data_valid.columns: covs_final.append("spike_ratio_FD_c")

        # --- STEP 2: REGRESSION (LI vs SLRT & SLRT_z) ---
        for target_var in ["SLRT", "SLRT_z"]:
            if target_var in data_valid.columns and data_valid[target_var].notna().sum() > 5:
                model, N = run_regression(data_valid, "LI", [target_var] + covs_final)
                if model and target_var in model.pvalues:
                    p_val = model.pvalues[target_var]
                    results.append({
                        "Contrast": contrast, "ROI": roi_name, "Test": f"Regress_{target_var}",
                        "Beta": model.params[target_var], "P": p_val, "N": N,
                        "Sig": "*" if p_val < 0.05 else ""
                    })
                    
                    plt.figure(figsize=(5,4))
                    color = "darkorange" if roi_name == "WholeBrain" else "purple"
                    sns.regplot(data=data_valid, x=target_var, y="LI", color=color)
                    plt.axhline(0, color='k', linestyle='-')
                    
                    title_name = "SLRT (Percentile)" if target_var == "SLRT" else "SLRT (Z-score)"
                    plt.title(f"{roi_name} LI vs {title_name}\np={p_val:.3f}")
                    plt.ylim(-1.1, 1.1)
                    plt.ylabel("LI")
                    plt.xlabel(title_name)
                    plt.tight_layout()
                    plt.savefig(con_dir / f"Reg_{target_var}_{roi_name}.png")
                    plt.close()

        # --- STEP 3: GROUP DIFFERENCE ---
        if data_valid["Group_Bin"].nunique() > 1:
            model, N = run_regression(data_valid, "LI", ["Group_Bin"] + covs_final)
            if model and "Group_Bin" in model.pvalues:
                p_val = model.pvalues["Group_Bin"]
                results.append({
                    "Contrast": contrast, "ROI": roi_name, "Test": "Group_Diff",
                    "Beta": model.params["Group_Bin"], "P": p_val, "N": N,
                    "Sig": "*" if p_val < 0.05 else ""
                })
                
                plt.figure(figsize=(4,4))
                pal = "Oranges" if roi_name == "WholeBrain" else "Purples"
                sns.boxplot(data=data_valid, x="Group", y="LI", palette=pal)
                sns.stripplot(data=data_valid, x="Group", y="LI", color="k", alpha=0.5)
                plt.axhline(0, color='k', linestyle='-')
                plt.title(f"{roi_name} LI Group Diff\np={p_val:.3f}")
                plt.ylim(-1.1, 1.1)
                plt.tight_layout()
                plt.savefig(con_dir / f"GroupDiff_{roi_name}.png")
                plt.close()

# Save Results
df_res = pd.DataFrame(results)
if not df_res.empty:
    cols = ["Contrast", "ROI", "Test", "Beta", "P", "Sig", "N"]
    final_cols = [c for c in cols if c in df_res.columns]
    df_res = df_res[final_cols].sort_values(["Contrast", "ROI", "Test"])
    
    out_file = OUTPUT_ROOT / "H3_Results_OneSample_and_Regression.csv"
    df_res.to_csv(out_file, index=False)
    print(f"\n>>> Results saved to: {out_file}")
    print(df_res.head(40).to_string())
else:
    print("No significant results.")

print("\n>>> Done.")
