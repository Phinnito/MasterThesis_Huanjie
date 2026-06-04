#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:03:56 2025

@author: phinnito
"""
#############################################################
##  Input    ################################################
#############################################################
#############################################################

from pathlib import Path
import pandas as pd, numpy as np
import io
import re
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
import matplotlib.pyplot as plt
import warnings

# ---- 1) BASIC STUDY INFO (edit these) ----
BIDS_ROOT = Path("/your/path/")
#FMRIPREP_ROOT = BIDS_ROOT / "derivatives"
FMRIPREP_ROOT = BIDS_ROOT / "derivatives" / "fmriprep"
LOG_ROOT = BIDS_ROOT / "log_files"

SUBJECT = "sub-5084"
SESSION = "ses-1"
TASK = "vstring"
SPACE = "MNIPediatricAsym_cohort-3_res-2"   

func_dir = FMRIPREP_ROOT / SUBJECT / SESSION / "func"

bold_img = func_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_run-1_space-{SPACE}_desc-preproc_bold.nii.gz"
conf_tsv = func_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_run-1_desc-confounds_timeseries.tsv"
mask_img = func_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_run-1_space-{SPACE}_desc-brain_mask.nii.gz"

log_dir = LOG_ROOT / SUBJECT / SESSION
log_file = next(log_dir.glob("*.log"))

output_dir = BIDS_ROOT / "Nilearn" / "nilearn_glm_level1_new" / SUBJECT / SESSION

print("[INPUTS]")
print(" bold :", bold_img)
print(" conf :", conf_tsv)
print(" mask :", mask_img)
print(" log  :", log_file)



#############################################################
##  Analysing log files (SIMPLIFIED)  ######################
#############################################################
#############################################################

# Picture codes for main conditions
PICTURE_MAIN    = {70: "word", 80: "pseudoword", 90: "falsefont"}
PICTURE_TARGET  = {75: "word", 85: "pseudoword", 95: "falsefont"}
IGNORE_PICTURES = {10, 50, 100, 190, 200, 255, 0}
# Updated: All button press codes
RESP_CODES      = {1, 61, 2, 62, 3, 63, 4, 64}  
PULSE_CODE      = 199

def _read_presentation_table(log_path: str | Path) -> pd.DataFrame:
    """Read Presentation log file and return cleaned dataframe."""
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    hdr_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Subject") and "\tEvent Type\t" in ln),
        None
    )
    end_idx = next(
        (j for j in range(hdr_idx + 1, len(lines)) if lines[j].startswith("Event Type") and "\tCode\t" in lines[j]),
        None
    )
    data_block = "\n".join(lines[hdr_idx: end_idx])
    df = pd.read_csv(io.StringIO(data_block), sep="\t")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    keep = ["Event Type", "Code", "Time", "Duration", "Stim Type"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA
    for k in ["Code", "Time", "Duration"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df[keep].dropna(subset=["Event Type", "Code", "Time"])

def build_events_for_glm_from_log_simplified(
    log_path: str | Path,
    modeling: str = "block",          
    use_log_duration: bool = False,
    default_duration: float = 0.66,
    drop_pre_pulse: bool = True,
    block_gap: float = 3.0,           
):
    """
    Simplified version: Only extract word/pseudoword/falsefont blocks
    No accuracy tracking needed
    """
    # 1) Read and time-align to first Pulse
    df = _read_presentation_table(log_path)
    has_pulse = (df["Event Type"].eq("Pulse") & (df["Code"] == PULSE_CODE)).any()
    t0_ms = df.loc[df["Event Type"].eq("Pulse") & (df["Code"] == PULSE_CODE), "Time"].min() if has_pulse else df["Time"].min()
    if drop_pre_pulse and has_pulse:
        df = df[df["Time"] >= t0_ms].copy()

    # 2) Extract main condition events only
    rows = []
    pic = df[df["Event Type"].eq("Picture")].copy().sort_values("Time")
    for _, r in pic.iterrows():
        code = int(r["Code"]) if pd.notna(r["Code"]) else None
        time_ms = float(r["Time"])
        dur_s = (float(r["Duration"]) / 10000.0) if (use_log_duration and pd.notna(r["Duration"])) else float(default_duration)
        onset_s = (time_ms - t0_ms) / 10000.0

        # Only keep main conditions (70/80/90) and targets
        if code in PICTURE_MAIN:
            base_name = PICTURE_MAIN[code]
            rows.append({"onset": onset_s, "duration": dur_s, "trial_type": base_name})
        elif code in PICTURE_TARGET:
            # Treat targets as part of their base category
            base_name = PICTURE_TARGET[code]
            rows.append({"onset": onset_s, "duration": dur_s, "trial_type": base_name})

    # Button presses as nuisance regressor (all codes included)
    resp = df[df["Event Type"].eq("Response") & df["Code"].isin(RESP_CODES)]
    for _, r in resp.iterrows():
        rows.append({"onset": (float(r["Time"]) - t0_ms) / 10000.0, "duration": 0.0, "trial_type": "button"})

    events_trial = pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)
    if events_trial.empty:
        raise ValueError(f"{log_path}: empty events after parsing")

    # 3) If block modeling, collapse consecutive same-condition trials
    if modeling.lower() == "block":
        main_set = {"word", "pseudoword", "falsefont"}
        
        # --- START: NEW, MORE ROBUST BLOCK-MERGING LOGIC ---
        
        # Seperate main events（word, pseudo, ff）from（button）
        main_events = events_trial[events_trial["trial_type"].isin(main_set)].copy()
        nuisance_events = events_trial[~events_trial["trial_type"].isin(main_set)].copy()

        if not main_events.empty:
            # create a unique 'blockID'  
            # check the current block if it's same as last block
            main_events['block_id'] = (main_events['trial_type'] != main_events['trial_type'].shift()).cumsum()
            
            # calculate the duration time and start time of BLOCK 
            blocks = main_events.groupby(['trial_type', 'block_id']).agg(
                # onset should be the min value of each trails in BLO
                onset=('onset', 'min'),
                # BLOCK duration = Last onset + last duration - first onset
                duration=('onset', lambda x: x.max() - x.min() + main_events.loc[x.idxmax(), 'duration'])
            ).reset_index()
            
            # Abandon "Block ID"
            blocks = blocks.drop(columns='block_id')
            
            # combine new block and button
            events_glm = pd.concat([blocks, nuisance_events], ignore_index=True).sort_values("onset").reset_index(drop=True)
            
        else: 
            events_glm = nuisance_events

        # --- END: NEW, MORE ROBUST BLOCK-MERGING LOGIC ---

    else:
        events_glm = events_trial

    # 4) Info dict
    info = {
        "pulse_found": bool(has_pulse),
        "t0_sec": t0_ms / 10000.0,
        "n_events": len(events_glm),
        "trial_types": list(events_glm["trial_type"].unique()),
        "counts_by_type": events_glm["trial_type"].value_counts().to_dict(),
        "modeling": modeling,
    }
    return events_glm, info

# Build events
events_glm, info = build_events_for_glm_from_log_simplified(
    log_file,
    modeling="block",        
    use_log_duration=False,
    default_duration=0.66,
    drop_pre_pulse=True,
    block_gap=3.0
)

print("\n[LOG ANALYSIS RESULTS]")
print("Trial types found:", info["trial_types"])
print("Counts by type:", info["counts_by_type"])

#############################################################
####  Confounds (SIMPLIFIED - MOTION6 ONLY) ################
#############################################################
#############################################################

FD_THRESH   = 1.5   # mm
N_ACOMPCOR  = 6     # number of aCompCor components
DEMEAN      = True  

# Only use 6 motion parameters
MOTION6 = ["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z"]

def pick_motion6(c: pd.DataFrame) -> pd.DataFrame:
    """Get only the 6 basic motion parameters."""
    if not all(col in c.columns for col in MOTION6):
        missing = [col for col in MOTION6 if col not in c.columns]
        raise ValueError(f"Missing motion columns: {missing}")
    return c[MOTION6].astype(float).copy()

def pick_acompcor(c: pd.DataFrame, n: int) -> pd.DataFrame | None:
    """Pick first n aCompCor columns if available."""
    if n <= 0: 
        return None
    pat = re.compile(r"^(aCompCor\d+|a_comp_cor_\d+)$")
    cols = sorted([col for col in c.columns if pat.match(col)])[:n]
    return c[cols].astype(float).copy() if cols else None

# Read confounds
c = pd.read_csv(conf_tsv, sep="\t")
n_tp = len(c)

# Get motion6 only
motion_block = pick_motion6(c)

# FD spikes for scrubbing
spike_cols = []
spikes_df = None
if "framewise_displacement" in c.columns:
    fd = c["framewise_displacement"].astype(float).fillna(0.0).to_numpy()
    idx = np.where(fd > float(FD_THRESH))[0]
    if idx.size > 0:
        spikes = np.zeros((n_tp, idx.size), float)
        for j, i in enumerate(idx):
            spikes[i, j] = 1.0
        spike_cols = [f"spike_FD_{i:04d}" for i in idx]
        spikes_df = pd.DataFrame(spikes, columns=spike_cols)

# Non-steady-state outliers
nss_cols = [col for col in c.columns if col.startswith("non_steady_state_outlier")]
nss_df = c[nss_cols].fillna(0.0).astype(float).copy() if nss_cols else None

# aCompCor
acompcor_df = pick_acompcor(c, N_ACOMPCOR)
acompcor_cols = list(acompcor_df.columns) if acompcor_df is not None else []

# Concatenate all confounds
parts = [motion_block]
for p in (spikes_df, nss_df, acompcor_df):
    if p is not None and not p.empty:
        parts.append(p)
confounds_df = pd.concat(parts, axis=1)

# Demean continuous columns
if DEMEAN and not confounds_df.empty:
    protect = set(spike_cols + nss_cols)
    cont_cols = [col for col in confounds_df.columns if col not in protect]
    if cont_cols:
        confounds_df[cont_cols] = confounds_df[cont_cols] - confounds_df[cont_cols].mean(axis=0)

print("\n[CONFOUNDS]")
print("Shape:", confounds_df.shape)
print("Motion parameters: 6 (trans_x/y/z, rot_x/y/z)")
print(f"FD spikes: {len(spike_cols)}, NSS: {len(nss_cols)}, aCompCor: {len(acompcor_cols)}")

#############################################################
#############################################################
#############################################################
############### Testing Hypothesis H1 ######################
#############################################################
#############################################################
#############################################################

from nilearn.image import load_img

# Create output directory
h1_dir = output_dir / "H1"
h1_dir.mkdir(parents=True, exist_ok=True)
print(f"\n[OUTPUT DIRECTORY] {h1_dir}")

# TR from header
TR = 1.98378

# Clean events and confounds for GLM
def clean_for_glm(events_in, conf_in, bold_path, TR):
    """
    Ensure events and confounds are properly aligned with scan duration.
    (Cleaned version: No debug prints)
    """

    n_scans = load_img(str(bold_path)).shape[-1]
    run_end = float(n_scans * TR)
    
    # 2. Events cleaning
    evt = events_in.copy()
    
    evt = evt[np.isfinite(evt["onset"]) & np.isfinite(evt["duration"])].copy()
    evt = evt[evt["duration"] > 0].copy()
    
    # Clip events extending beyond run duration
    max_dur = (run_end - evt["onset"]).clip(lower=0)
    evt.loc[:, "duration"] = np.minimum(evt["duration"], max_dur)
    evt = evt[evt["duration"] > 0].copy()
    
    # 3. Confounds cleaning
    conf = conf_in.copy()
    conf = conf.apply(pd.to_numeric, errors="coerce")
    conf.replace([np.inf, -np.inf], np.nan, inplace=True)
    conf.fillna(0.0, inplace=True)
    
    # 4.Align confounds length
    if len(conf) != n_scans:
        print(f"[WARN] Aligning confounds: {len(conf)} rows -> {n_scans} scans")
        
        if len(conf) > n_scans:
            conf = conf.iloc[:n_scans].reset_index(drop=True)
        else:
            pad = pd.DataFrame(np.zeros((n_scans - len(conf), conf.shape[1])), columns=conf.columns)
            conf = pd.concat([conf, pad], axis=0, ignore_index=True)
    
    return evt, conf, n_scans

# Clean data
events_clean, confounds_clean, n_scans = clean_for_glm(events_glm, confounds_df, bold_img, TR)

# Remove problematic confound columns if they cause rank deficiency
#old solutions to some idiot problems = =
#conf_to_remove = confounds_clean.columns[confounds_clean.columns.str.startswith("spike_FD_")]
#if len(conf_to_remove) > 0:
#    print(f"[INFO] Removing {len(conf_to_remove)} FD spike columns to avoid rank deficiency")
#    confounds_clean = confounds_clean.drop(columns=conf_to_remove)

print("\n[DEBUG] Final event counts before GLM fitting:")
print(events_clean['trial_type'].value_counts())

# Fit FirstLevelModel
print("\n[FITTING GLM]")
first_level_model = FirstLevelModel(
    t_r=TR,
    slice_time_ref=0.5,
    hrf_model="spm", #####
    drift_model="cosine",
    high_pass=1/128,
    smoothing_fwhm=6.0,  # 6
    noise_model="ar1",
    mask_img=mask_img,
    standardize=False,
    minimize_memory=True,
    verbose=1
)

# Fit the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    first_level_model.fit(bold_img, events=events_clean, confounds=confounds_clean)

print("[GLM FIT COMPLETE]")

# Define contrasts for H1
dm = first_level_model.design_matrices_[0]
cols = list(dm.columns)

def find_regressor(name):
    """Find regressor column, ignoring derivatives."""
    for col in cols:
        if col.startswith(name) and "derivative" not in col.lower():
            return col
    return None

# Find the actual column names
word_col = find_regressor("word")
pseudo_col = find_regressor("pseudoword")
ff_col = find_regressor("falsefont")

print("[REGRESSORS FOUND]")
print(f"Word: {word_col}")
print(f"Pseudoword: {pseudo_col}")
print(f"Falsefont: {ff_col}")

# Create contrast vectors
def make_contrast_vector(positive_cols, negative_cols):
    """Create a contrast vector."""
    vec = np.zeros(len(cols))
    for col in positive_cols:
        if col and col in cols:
            vec[cols.index(col)] = 1.0 / len(positive_cols)
    for col in negative_cols:
        if col and col in cols:
            vec[cols.index(col)] = -1.0 / len(negative_cols)
    return vec

# Define contrasts for H1
contrasts = {}

# Main contrasts for H1
if word_col and ff_col:
    contrasts["word_gt_falsefont"] = make_contrast_vector([word_col], [ff_col])
    
if pseudo_col and ff_col:
    contrasts["pseudoword_gt_falsefont"] = make_contrast_vector([pseudo_col], [ff_col])

# Additional contrasts for exploration
if word_col and pseudo_col:
    contrasts["word_gt_pseudoword"] = make_contrast_vector([word_col], [pseudo_col])
    contrasts["pseudoword_gt_word"] = make_contrast_vector([pseudo_col], [word_col])

# Single condition betas
for name, col in [("word", word_col), ("pseudoword", pseudo_col), ("falsefont", ff_col)]:
    if col:
        contrasts[name] = make_contrast_vector([col], [])

print(f"\n[CONTRASTS DEFINED] {list(contrasts.keys())}")

# Compute and save contrast maps
print("\n[COMPUTING CONTRAST MAPS]")
contrast_maps = {}
for contrast_name, contrast_vec in contrasts.items():
    print(f"  Computing: {contrast_name}")
    
    # Compute maps
    z_map = first_level_model.compute_contrast(contrast_vec, output_type="z_score")
    t_map = first_level_model.compute_contrast(contrast_vec, output_type="stat")
    beta_map = first_level_model.compute_contrast(contrast_vec, output_type="effect_size")
    var_map = first_level_model.compute_contrast(contrast_vec, output_type="effect_variance")
    
    # Save maps
    z_path = h1_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_contrast-{contrast_name}_zmap.nii.gz"
    t_path = h1_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_contrast-{contrast_name}_tmap.nii.gz"
    z_map.to_filename(z_path)
    t_map.to_filename(t_path)
    
    beta_path = h1_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_contrast-{contrast_name}_betamap.nii.gz"
    beta_map.to_filename(beta_path)
    var_path = h1_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_contrast-{contrast_name}_varmap.nii.gz"
    var_map.to_filename(var_path)
    
    contrast_maps[contrast_name] = {
        "z_map": z_map,
        "t_map": t_map,
        "beta_map": beta_map,
        "var_map": var_map
    }


# Save design matrix visualization
print("\n[SAVING DESIGN MATRIX]")
dm_plot = plotting.plot_design_matrix(dm)
dm_path = h1_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_design_matrix.png"
dm_plot.get_figure().savefig(dm_path, dpi=300, bbox_inches='tight')
plt.close(dm_plot.get_figure())






# Create whole-brain visualization for key contrasts
print("\n[Creating whole-brain visualizations]")

for contrast_name in ["word_gt_falsefont", "pseudoword_gt_falsefont"]:
    if contrast_name in contrast_maps:
        # Glass brain
        display = plotting.plot_glass_brain(
            contrast_maps[contrast_name]["z_map"],
            threshold=2.3,
            colorbar=True,
            title=f'{SUBJECT} - {contrast_name.replace("_", " ").title()}'
        )
        glass_path = h1_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_{contrast_name}_glassbrain.png"
        display.savefig(glass_path, dpi=200)
        display.close()
        
        # Slice display
        display = plotting.plot_stat_map(
            contrast_maps[contrast_name]["z_map"],
            threshold=1.96,
            display_mode='z',
            cut_coords=8,
            colorbar=True,
            title=f'{SUBJECT} - {contrast_name.replace("_", " ").title()}'
        )
        slice_path = h1_dir / f"{SUBJECT}_{SESSION}_task-{TASK}_{contrast_name}_slices.png"
        display.savefig(slice_path, dpi=200)
        display.close()



print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"All outputs saved to: {h1_dir}")
print("\nKey outputs:")
print(f"  - Contrast maps: {len(contrast_maps)} contrasts")
print("  - Visualizations: design matrix,  whole-brain maps")
print("  - Data files: JSON results, CSV summary")





warnings.filterwarnings('ignore')


#############################################################
#############################################################
##  HYPOTHESIS 2 - LEXICALITY EFFECT  ######################
#############################################################
#############################################################


print("\n" + "="*60)
print("HYPOTHESIS 2 - COMPUTING LEXICALITY CONTRAST")
print("="*60)

h2_dir = output_dir / "H2"
h2_dir.mkdir(exist_ok=True)

contrast_name = "pseudoword_gt_word"

if pseudo_col and word_col:
    print(f"Computing {contrast_name}...")
    
    # 1. contrast vector
    lex_vec = make_contrast_vector([pseudo_col], [word_col])
    
    # 2. compute
    z_map = first_level_model.compute_contrast(lex_vec, output_type="z_score")
    
    # 3. save
    save_path = h2_dir / f"{SUBJECT}_{SESSION}_contrast-{contrast_name}_zmap.nii.gz"
    z_map.to_filename(save_path)
    print(f"Saved: {save_path}")
    
    plotting.plot_glass_brain(
        z_map, 
        threshold=2.3, 
        colorbar=True, 
        title=f"Lexicality Effect (Pseudo > Word): {SUBJECT}",
        output_file=h2_dir / f"{SUBJECT}_H2_GlassBrain.png"
    )

print("H2 Complete.")


#############################################################
#############################################################
##  HYPOTHESIS 3 - LATERALIZATION  ##########################
#############################################################
#############################################################


print("\n" + "="*60)
print("HYPOTHESIS 3 - WHOLE BRAIN LATERALITY INDEX (LI)")
print("Using Contrast Subtraction Method to mitigate Motor Artifacts")
print("="*60)

# Setup H3 output directory
h3_dir = output_dir / "H3"
h3_dir.mkdir(exist_ok=True)
print(f"\n[H3 OUTPUT DIRECTORY] {h3_dir}")

from nilearn.image import math_img

# ---------------------------------------------------------
# 1. Create Left and Right Hemisphere Masks
# ---------------------------------------------------------

print("\n[CREATING HEMISPHERE MASKS]")
from nilearn.image import new_img_like

# Load the brain mask
ref_img = load_img(mask_img)
ref_data = ref_img.get_fdata()
affine = ref_img.affine

# Create a grid of indices for every voxel (i, j, k)
indices = np.indices(ref_data.shape)

# Convert voxel indices to World Coordinates (X, Y, Z) using the affine matrix
# x_coords = M[0,0]*i + M[0,1]*j + M[0,2]*k + M[0,3]
x_coords = (indices[0] * affine[0, 0] + 
            indices[1] * affine[0, 1] + 
            indices[2] * affine[0, 2] + 
            affine[0, 3])

# Define Left Mask: x < -2 (buffer) AND inside brain mask
# Define Right Mask: x > 2 (buffer) AND inside brain mask
left_mask_data = (x_coords < -2) & (ref_data > 0)
right_mask_data = (x_coords > 2) & (ref_data > 0)

# Convert back to Nifti images so the rest of your script works
left_hemi_mask = new_img_like(ref_img, left_mask_data.astype(int))
right_hemi_mask = new_img_like(ref_img, right_mask_data.astype(int))

# Save masks for inspection (Optional but good for QC)
left_mask_path = h3_dir / f"{SUBJECT}_hemi-L_mask.nii.gz"
right_mask_path = h3_dir / f"{SUBJECT}_hemi-R_mask.nii.gz"
left_hemi_mask.to_filename(left_mask_path)
right_hemi_mask.to_filename(right_mask_path)

print("  Hemisphere masks created and saved.")

# ---------------------------------------------------------
# 2. Calculate Whole-Brain LI for Key Contrasts
# ---------------------------------------------------------
# We use the contrasts that substract "FalseFont" or "Pseudoword" 
# to cancel out the motor activation naturally.

target_contrasts = [
    "word_gt_falsefont",       # Global Reading Network (vOT dominant)
    "pseudoword_gt_falsefont", # Decoding Network (TPJ dominant)
    "word_gt_pseudoword",       # Semantic Network (IFG dominant)
    "pseudoword_gt_word"       # (Decoding Focus)
]

# LI Calculation Function
def calculate_weighted_li(z_map, mask_l, mask_r, threshold=0):
    """
    Calculates LI based on the sum of positive Z-scores (magnitude weighted).
    Formula: (Sum_L - Sum_R) / (Sum_L + Sum_R)
    Only considers voxels with Z > threshold.
    """
    from nilearn.masking import apply_mask
    
    # Extract data from left and right hemispheres
    # apply_mask returns a 1D array of voxel values inside the mask
    vals_l = apply_mask(z_map, mask_l)
    vals_r = apply_mask(z_map, mask_r)
    
    # Filter: keep only positive values (or above threshold)
    # Negative deactivations can mess up LI calculations
    vals_l = vals_l[vals_l > threshold]
    vals_r = vals_r[vals_r > threshold]
    
    # Calculate Sum of activation (Integral)
    sum_l = np.sum(vals_l) if len(vals_l) > 0 else 0
    sum_r = np.sum(vals_r) if len(vals_r) > 0 else 0
    
    # Compute LI
    if (sum_l + sum_r) == 0:
        li = 0
    else:
        li = (sum_l - sum_r) / (sum_l + sum_r)
        
    return li, sum_l, sum_r

print("\n[CALCULATING LI]")
li_results = []

# Threshold for including a voxel in LI calculation. 
# 0 means "all positive signal". 1.64 means "only significant signal (p<0.05)"
LI_THRESHOLD = 0 

for contrast in target_contrasts:
    if contrast in contrast_maps:
        print(f"  Analyzing: {contrast}")
        z_map = contrast_maps[contrast]["z_map"]
        
        li_val, sum_l, sum_r = calculate_weighted_li(
            z_map, left_hemi_mask, right_hemi_mask, threshold=LI_THRESHOLD
        )
        
        result_entry = {
            "Contrast": contrast,
            "LI": li_val,
            "Sum_Left_Z": sum_l,
            "Sum_Right_Z": sum_r
        }
        li_results.append(result_entry)
        print(f"    -> LI = {li_val:.3f} (L_sum={sum_l:.1f}, R_sum={sum_r:.1f})")
    else:
        print(f"  [WARN] Contrast {contrast} not found in maps.")

# Save Results to CSV
if li_results:
    df_li = pd.DataFrame(li_results)
    csv_path = h3_dir / f"{SUBJECT}_{SESSION}_WholeBrain_LI.csv"
    df_li.to_csv(csv_path, index=False)
    print(f"\n  LI data saved to: {csv_path}")

# ---------------------------------------------------------
# 3. H3 Visualization
# ---------------------------------------------------------
print("\n[CREATING H3 VISUALIZATIONS]")

if li_results:
    # A. Plotting the Masks (Quality Control)
    # ---------------------------------------
    # Show L and R masks on the glass brain to confirm split
    from nilearn import plotting
    
    # Create a composite mask image for visualization: 1=Left, 2=Right
    # We use math_img to combine them
    combo_mask = math_img("img1 + 2*img2", img1=left_hemi_mask, img2=right_hemi_mask)
    
    viz_mask_path = h3_dir / f"{SUBJECT}_Hemisphere_Split_QC.png"
    display = plotting.plot_roi(
        combo_mask, 
        title=f"{SUBJECT} Hemisphere Split (Left=Red, Right=Blue)",
        display_mode='z', 
        cut_coords=5,
        cmap='Paired'
    )
    display.savefig(viz_mask_path, dpi=200)
    display.close()
    
    # B. Bar Chart of LI Values
    # ---------------------------------------
    df_viz = pd.DataFrame(li_results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors based on LI value (Left=Blue/Positive, Right=Red/Negative)
    colors = ['#1f77b4' if x >= 0 else '#d62728' for x in df_viz['LI']]
    
    bars = ax.bar(df_viz['Contrast'], df_viz['LI'], color=colors, alpha=0.8, width=0.6)
    
    # Styling
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel('Laterality Index (LI)\n<- Right      Left ->', fontsize=12)
    ax.set_title(f'Whole-Brain Laterality Index\n(Weighted Sum of Z > {LI_THRESHOLD})', fontsize=14, fontweight='bold')
    
    # Add threshold lines for "Strong Lateralization" (usually +/- 0.2)
    ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(df_viz)-0.6, 0.22, 'Left Dominance', fontsize=8, color='gray')
    ax.text(len(df_viz)-0.6, -0.25, 'Right Dominance', fontsize=8, color='gray')
    
    # Label values
    for bar in bars:
        height = bar.get_height()
        label_y = height + 0.05 if height >= 0 else height - 0.1
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    # Clean x-axis labels
    clean_labels = [l.replace('_gt_', ' > ').replace('_', ' ').title() for l in df_viz['Contrast']]
    ax.set_xticklabels(clean_labels, rotation=15, ha='center')
    
    plt.tight_layout()
    viz_bar_path = h3_dir / f"{SUBJECT}_{SESSION}_WholeBrain_LI_Chart.png"
    plt.savefig(viz_bar_path, dpi=300)
    plt.close()
    print(f"  LI Chart saved to: {viz_bar_path}")

print("\n" + "="*60)
print("H3 ANALYSIS COMPLETE")
print("="*60)



print("[H3 ANALYSIS COMPLETE]")
print("Now please run the next subject : )")




