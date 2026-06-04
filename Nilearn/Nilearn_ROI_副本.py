#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-Based Cluster Analysis Pipeline
Strictly separate ROI lists for H1, H2, and H3.

Created: January 2026
"""

from pathlib import Path
import pandas as pd
import numpy as np
from nilearn.image import load_img, get_data, math_img, iter_img, resample_to_img, new_img_like
from nilearn.regions import connected_regions
import warnings

warnings.filterwarnings("ignore")

#############################################################
##  CONFIGURATION (STRICT DEFINITIONS)
#############################################################

# Base directories
BIDS_ROOT = Path("/your/path/")
GLM_OUTPUT_DIR = BIDS_ROOT / "Nilearn" / "nilearn_glm_level1_new"
SUBJECT_LIST = BIDS_ROOT / "Nilearn" / "Indivi" / "copied_subjects_list.csv"

# Output directory
OUTPUT_ROOT = BIDS_ROOT / "Nilearn" / "ROI_Cluster_Analysis_2.5"
TASK = "vstring"

# Analysis Parameters
T_THRESHOLD = 2.5
MIN_CLUSTER_SIZE_MM3 = 10 

# --- ROI DEFINITIONS (STRICT) ---

# 1. Master list of file paths (Loads everything into memory)
MASK_PATHS = {
    'L_VWFA1': BIDS_ROOT / "Nilearn" / "ROI_Mask" / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_L_VWFA1.nii.gz",
    'L_VWFA2': BIDS_ROOT / "Nilearn" / "ROI_Mask" / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_L_VWFA2.nii.gz",
    'L_vOTC':  BIDS_ROOT / "Nilearn" / "ROI_Mask" / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_L_vOTC.nii.gz",
    'L_IFG':   BIDS_ROOT / "Nilearn" / "ROI_Mask" / "HO_IFG_L_thr25_MNIPed.nii.gz",
    'L_STG':   BIDS_ROOT / "Nilearn" / "ROI_Mask" / "HO_STG_L_thr25_MNIPed.nii.gz",
    
    'R_VWFA1': BIDS_ROOT / "Nilearn" / "ROI_Mask" / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_R_VWFA1.nii.gz",
    'R_VWFA2': BIDS_ROOT / "Nilearn" / "ROI_Mask" / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_R_VWFA2.nii.gz",
    'R_vOTC':  BIDS_ROOT / "Nilearn" / "ROI_Mask" / "vOTCcore_HO_FGITG_IOG_pTO_thr25p0_space-MNIPediatricAsym_cohort-3_res-2_R_vOTC.nii.gz",
    'R_IFG':   BIDS_ROOT / "Nilearn" / "ROI_Mask" / "HO_IFG_R_thr25_MNIPed.nii.gz",
    'R_STG':   BIDS_ROOT / "Nilearn" / "ROI_Mask" / "HO_STG_R_thr25_MNIPed.nii.gz",
}

# 2. Hypothesis Specific Lists (This controls the loops)

# H1: Activation check (Only VWFA/vOTC regions)
H1_CONTRASTS = ["word_gt_falsefont", "pseudoword_gt_falsefont"]
H1_MASKS = ["L_VWFA1", "L_VWFA2", "L_vOTC"]

# H2: Lexicality effect (Includes IFG/STG + VWFA/vOTC)
H2_CONTRASTS = ["word_gt_pseudoword", "pseudoword_gt_word"]
H2_MASKS = ["L_IFG", "L_STG", "L_VWFA1", "L_VWFA2", "L_vOTC"]

# H3: Laterality (Pairs)
H3_CONTRASTS = ["word_gt_falsefont", "pseudoword_gt_falsefont", "word_gt_pseudoword", "pseudoword_gt_word"]
H3_MASK_PAIRS = [
    ("L_IFG", "R_IFG"),
    ("L_STG", "R_STG"),
    ("L_VWFA1", "R_VWFA1"),
    ("L_VWFA2", "R_VWFA2"),
    ("L_vOTC", "R_vOTC")
]

#############################################################
##  HELPER FUNCTIONS
#############################################################

def find_contrast_maps(subject_dir, session, task, contrast_name):
    # Adjust this path if your H1/H2 folders structure is different
    # Assuming standard GLM output structure
    search_dir = subject_dir / "H1" 
    if not search_dir.exists():
        # Fallback to try H2 if H1 doesn't exist, or just root subject dir
        search_dir = subject_dir 
    
    pattern_base = f"*task-{task}_contrast-{contrast_name}_"
    t_map_files = list(search_dir.glob(f"{pattern_base}tmap.nii.gz"))
    
    if not t_map_files:
        return None
    return t_map_files[0]

def extract_clusters(t_map, mask_img, threshold, min_size=MIN_CLUSTER_SIZE_MM3):
    """
    Returns a list of dictionaries, one for each cluster found.
    """
    # 1. Mask and Threshold
    mask_res = resample_to_img(mask_img, t_map, interpolation='nearest')
    
    # Logic: Keep voxel IF (in mask) AND (value >= threshold)
    expression = f'img * (mask > 0) * (img >= {threshold})'
    thresh_img = math_img(expression, img=t_map, mask=mask_res)
    
    # Quick check if empty
    if np.sum(get_data(thresh_img)) == 0:
        return [] # No activation

    # 2. Find Connected Components
    try:
        regions_img, _ = connected_regions(thresh_img, min_region_size=min_size, extract_type='connected_components')
    except Exception:
        return []

    # 3. Process each cluster
    clusters = []
    # If regions_img is 3D (1 cluster), iter_img fails, so we handle that:
    if len(regions_img.shape) == 3:
        iter_list = [regions_img]
    else:
        iter_list = iter_img(regions_img)

    for i, region in enumerate(iter_list):
        dat = get_data(region)
        valid_voxels = dat[dat != 0]
        if len(valid_voxels) == 0: continue
        
        clusters.append({
            'id': i + 1,
            'size': len(valid_voxels),
            'mean': float(np.mean(valid_voxels)),
            'peak': float(np.max(valid_voxels)),
            'img': region
        })
        
    return clusters

def create_diff_map(t_map, mask_l, mask_r):
    """
    Creates (Left - Right) map.
    Positive = Left dominant. Negative = Right dominant.
    """
    ml_res = resample_to_img(mask_l, t_map, interpolation='nearest')
    mr_res = resample_to_img(mask_r, t_map, interpolation='nearest')
    
    dat = get_data(t_map)
    l_bool = get_data(ml_res).astype(bool)
    r_bool = get_data(mr_res).astype(bool)
    
    out_dat = np.zeros_like(dat)
    out_dat[l_bool] = dat[l_bool]       # Keep L positive
    out_dat[r_bool] = -dat[r_bool]      # Flip R to negative
    
    return new_img_like(t_map, out_dat)

#############################################################
##  PROCESSING PIPELINES (STRICT LOOPS)
#############################################################

def process_h1(subject, session, glm_dir, all_masks, out_dir):
    print(f"  [H1] Processing {len(H1_MASKS)} ROIs: {H1_MASKS}")
    results = []
    
    for contrast in H1_CONTRASTS:
        t_file = find_contrast_maps(glm_dir, session, TASK, contrast)
        if not t_file: continue
        t_img = load_img(t_file)
        
        # STRICT LOOP: Only iterate over H1_MASKS
        for mask_name in H1_MASKS:
            if mask_name not in all_masks: continue # Safety check
            
            # Extract
            clusters = extract_clusters(t_img, all_masks[mask_name], T_THRESHOLD)
            
            # Save & Record
            base_name = f"{subject}_{contrast}_{mask_name}"
            
            if not clusters:
                # Record empty
                results.append({'Subject': subject, 'Hypothesis': 'H1', 'Contrast': contrast, 
                                'ROI': mask_name, 'Cluster_ID': 0, 'Size': 0, 'Mean': 0})
            else:
                for c in clusters:
                    fname = f"{base_name}_cluster{c['id']:02d}.nii.gz"
                    c['img'].to_filename(out_dir / fname)
                    
                    results.append({
                        'Subject': subject, 'Hypothesis': 'H1', 'Contrast': contrast, 
                        'ROI': mask_name, 'Cluster_ID': c['id'], 
                        'Size': c['size'], 'Mean': c['mean'], 'Peak': c['peak'],
                        'File': fname
                    })
    return results

def process_h2(subject, session, glm_dir, all_masks, out_dir):
    print(f"  [H2] Processing {len(H2_MASKS)} ROIs: {H2_MASKS}")
    results = []
    
    for contrast in H2_CONTRASTS:
        t_file = find_contrast_maps(glm_dir, session, TASK, contrast)
        if not t_file: continue
        t_img = load_img(t_file)
        
        # STRICT LOOP: Only iterate over H2_MASKS
        for mask_name in H2_MASKS:
            if mask_name not in all_masks: continue
            
            clusters = extract_clusters(t_img, all_masks[mask_name], T_THRESHOLD)
            
            base_name = f"{subject}_{contrast}_{mask_name}"
            
            if not clusters:
                results.append({'Subject': subject, 'Hypothesis': 'H2', 'Contrast': contrast, 
                                'ROI': mask_name, 'Cluster_ID': 0, 'Size': 0, 'Mean': 0})
            else:
                for c in clusters:
                    fname = f"{base_name}_cluster{c['id']:02d}.nii.gz"
                    c['img'].to_filename(out_dir / fname)
                    results.append({
                        'Subject': subject, 'Hypothesis': 'H2', 'Contrast': contrast, 
                        'ROI': mask_name, 'Cluster_ID': c['id'], 
                        'Size': c['size'], 'Mean': c['mean'], 'Peak': c['peak'],
                        'File': fname
                    })
    return results

def process_h3(subject, session, glm_dir, all_masks, out_dir):
    print(f"  [H3] Processing {len(H3_MASK_PAIRS)} ROI Pairs")
    results = []
    
    for contrast in H3_CONTRASTS:
        t_file = find_contrast_maps(glm_dir, session, TASK, contrast)
        if not t_file: continue
        t_img = load_img(t_file)
        
        # STRICT LOOP: Only iterate over H3_MASK_PAIRS
        for l_name, r_name in H3_MASK_PAIRS:
            if l_name not in all_masks or r_name not in all_masks: continue
            
            roi_base = l_name.replace("L_", "")
            diff_map = create_diff_map(t_img, all_masks[l_name], all_masks[r_name])
            
            # Left Dominant (Positive > Threshold)
            clus_L = extract_clusters(diff_map, all_masks[l_name], T_THRESHOLD)
            
            # Right Dominant (Negative < -Threshold -> Inverse > Threshold)
            inv_diff = math_img("-img", img=diff_map)
            clus_R = extract_clusters(inv_diff, all_masks[r_name], T_THRESHOLD)
            
            # Save L
            for c in clus_L:
                fname = f"{subject}_{contrast}_{roi_base}_LEFT_cluster{c['id']:02d}.nii.gz"
                c['img'].to_filename(out_dir / fname)
                results.append({
                    'Subject': subject, 'Hypothesis': 'H3', 'Contrast': contrast, 
                    'ROI': roi_base, 'Dominance': 'Left',
                    'Cluster_ID': c['id'], 'Size': c['size'], 'Mean': c['mean'], 'File': fname
                })
                
            # Save R
            for c in clus_R:
                fname = f"{subject}_{contrast}_{roi_base}_RIGHT_cluster{c['id']:02d}.nii.gz"
                c['img'].to_filename(out_dir / fname)
                results.append({
                    'Subject': subject, 'Hypothesis': 'H3', 'Contrast': contrast, 
                    'ROI': roi_base, 'Dominance': 'Right',
                    'Cluster_ID': c['id'], 'Size': c['size'], 'Mean': c['mean'], 'File': fname
                })
                
            if not clus_L and not clus_R:
                results.append({'Subject': subject, 'Hypothesis': 'H3', 'Contrast': contrast, 
                                'ROI': roi_base, 'Dominance': 'None', 'Cluster_ID': 0, 'Size': 0})
                                
    return results

#############################################################
##  MAIN
#############################################################

def main():
    print("Loading Masks...")
    loaded_masks = {}
    for k, v in MASK_PATHS.items():
        if v.exists():
            loaded_masks[k] = load_img(v)
        else:
            print(f"Warning: {k} mask not found at {v}")

    df_sub = pd.read_csv(SUBJECT_LIST)
    all_data = []

    for _, row in df_sub.iterrows():
        sub = row['Subject']
        sess = row['Session']
        print(f"\nProcessing {sub}...")
        
        glm = GLM_OUTPUT_DIR / sub / sess
        if not glm.exists(): continue
        
        # H1
        h1_dir = OUTPUT_ROOT / sub / "H1"
        h1_dir.mkdir(parents=True, exist_ok=True)
        all_data.extend(process_h1(sub, sess, glm, loaded_masks, h1_dir))
        
        # H2
        h2_dir = OUTPUT_ROOT / sub / "H2"
        h2_dir.mkdir(parents=True, exist_ok=True)
        all_data.extend(process_h2(sub, sess, glm, loaded_masks, h2_dir))
        
        # H3
        h3_dir = OUTPUT_ROOT / sub / "H3"
        h3_dir.mkdir(parents=True, exist_ok=True)
        all_data.extend(process_h3(sub, sess, glm, loaded_masks, h3_dir))

    # Save Summary
    if all_data:
        df_out = pd.DataFrame(all_data)
        out_path = OUTPUT_ROOT / "Cluster_Summary_Strict.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\nDone. Summary saved to {out_path}")
    else:
        print("\nNo results found.")

if __name__ == "__main__":
    main()
