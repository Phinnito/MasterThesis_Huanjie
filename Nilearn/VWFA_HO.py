# -*- coding: utf-8 -*-
"""
Build bilateral vOTC mask (posterior/TO FG + ITG + ventral occipital)
from Harvard-Oxford atlas, resample to MNIPediatricAsym cohort-3 res-2,
threshold, then define VWFA1/VWFA2 masks as the intersection of vOTC
with two literature-based boxes (and their right-hemisphere mirrors).

Outputs (all in OUTROOT/ROI_Mask):
  - vOTCcore_<TAG>_thrXX_L_vOTC.nii.gz
  - vOTCcore_<TAG>_thrXX_R_vOTC.nii.gz
  - vOTCcore_<TAG>_thrXX_L_VWFA1.nii.gz
  - vOTCcore_<TAG>_thrXX_R_VWFA1.nii.gz
  - vOTCcore_<TAG>_thrXX_L_VWFA2.nii.gz
  - vOTCcore_<TAG>_thrXX_R_VWFA2.nii.gz
"""

import os
import re
import numpy as np
import nibabel as nib
from nilearn import datasets, image


# ----------------------------
# Helper functions
# ----------------------------

def norm_label(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_maybe_img(x):
    """Version-safe loader for nilearn atlas maps."""
    if isinstance(x, nib.spatialimages.SpatialImage):
        return x
    return nib.load(x)


def ho_label_to_volume_index(labels, label_idx):
    """
    HO cortical probabilistic atlas:
    Label list may have 'Background' at index 0, while 4D maps exclude it.
    If Background exists:
        volume index = label_idx - 1
    else:
        volume index = label_idx
    """
    labels = [norm_label(l) for l in labels]
    if labels and labels[0].lower().startswith("background"):
        return label_idx - 1
    return label_idx


def find_label_indices(labels, pattern_list):
    """
    Find label indices whose names match ANY regex pattern (case-insensitive).
    Returns indices in the labels list (not 4D volume indices).
    """
    labels = [norm_label(l) for l in labels]
    hits = []
    for i, name in enumerate(labels):
        lname = name.lower()
        for pat in pattern_list:
            if re.search(pat.lower(), lname):
                hits.append(i)
                break
    # remove background if present
    hits = [i for i in hits if not labels[i].lower().startswith("background")]
    return sorted(set(hits))


def make_coord_mask_like(img, z_max=None, y_min=None, y_max=None):
    """
    Optional trim using world coordinates.
    z_max: keep z < z_max
    y_min: keep y > y_min
    y_max: keep y < y_max
    """
    shape = img.shape[:3]
    affine = img.affine
    ijk = np.indices(shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(affine, ijk)

    m = np.ones((xyz.shape[0],), dtype=bool)
    if z_max is not None:
        m &= (xyz[:, 2] < z_max)
    if y_min is not None:
        m &= (xyz[:, 1] > y_min)
    if y_max is not None:
        m &= (xyz[:, 1] < y_max)

    return m.reshape(shape).astype(np.uint8)


def save_uint8_mask(data, ref_img, out_path):
    img = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, out_path)


def get_world_xyz(img):
    """Return flattened ijk and xyz arrays for the image grid."""
    shape = img.shape[:3]
    affine = img.affine
    ijk = np.indices(shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(affine, ijk)
    return ijk, xyz


def make_hemisphere_masks(img):
    """Return (left_mask, right_mask) based on world x < 0 / x > 0."""
    shape = img.shape[:3]
    affine = img.affine
    ijk = np.indices(shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(affine, ijk)
    left = (xyz[:, 0] < 0).reshape(shape)
    right = (xyz[:, 0] > 0).reshape(shape)
    return left.astype(np.uint8), right.astype(np.uint8)


def mirror_x_bounds(bounds):
    """Mirror a box's x-range from left to right hemisphere."""
    x_min, x_max = bounds["x"]
    x_min_r = -x_max
    x_max_r = -x_min
    return {
        "x": (x_min_r, x_max_r),
        "y": bounds["y"],
        "z": bounds["z"],
    }


# ============================
# CONFIG
# ============================

# 1) Target MNIPediatric template
TARGET_T1 = "/Volumes/BrainMap$/studies/lexi/mri/analyses/vstring_ik/derivatives/fmriprep/sub-5001/anat/sub-5001_run-1_space-MNIPediatricAsym_cohort-3_res-2_desc-preproc_T1w.nii.gz"

# 2) Output root folder (will create a ROI_Mask subfolder here)
OUTROOT = "/Users/phinnito/Downloads"

# 3) HO atlas name
HO_ATLAS_NAME = "cort-prob-2mm"

# 4) Probability threshold (0-100). More conservative from lit: 25 is common.
PROB_THR = 25.0

# 5) Optional coordinate trims（it depends, normally False）
APPLY_Z_CUTOFF = False
Z_MAX_MM = 10.0

APPLY_Y_MIN = False
Y_MIN_MM = -85.0

APPLY_Y_MAX = False
Y_MAX_MM = -30.0

# 6) VWFA boxes (LEFT hemisphere, world / MNI coordinates)
VWFA1_BOUNDS_L = {
    "x": (-58.0, -36.0),
    "y": (-84.0, -62.5),
    "z": (-25.0, -5.0),
}

VWFA2_BOUNDS_L = {
    "x": (-58.0, -36.0),
    "y": (-61.5, -40.0),
    "z": (-30.0, -8.0),
}

# 7) Naming tag
NAME_TAG = "HO_FGITG_IOG_pTO"

# ============================
# End CONFIG
# ============================


# ----------------------------
# 0) Basic checks & output folder
# ----------------------------

if not os.path.exists(TARGET_T1):
    raise FileNotFoundError(f"Target template not found:\n{TARGET_T1}")

OUTDIR = os.path.join(OUTROOT, "ROI_Mask")
os.makedirs(OUTDIR, exist_ok=True)

target_img = nib.load(TARGET_T1)


# ----------------------------
# 1) Build FG+ITG+ventral occipital prob map in HO space
# ----------------------------

ho = datasets.fetch_atlas_harvard_oxford(HO_ATLAS_NAME)
ho_img = load_maybe_img(ho.maps)
labels = [norm_label(l) for l in ho.labels]
ho_data = ho_img.get_fdata()

if ho_data.ndim != 4:
    raise ValueError(
        "Expected a 4D HO probabilistic atlas. "
        "Try HO_ATLAS_NAME = 'cort-prob-1mm' or 'cort-prob-2mm'."
    )

# posterior / temporal-occipital FG + ITG
fus_pats = [
    r"temporal fusiform cortex.*posterior",
    r"temporal occipital fusiform cortex"
]
itg_pats = [
    r"inferior temporal gyrus.*posterior",
    r"inferior temporal gyrus.*temporooccipital"
]

# Ventral occipital / IOG-approx:
#   - Lateral Occipital Cortex, inferior division
#   - Occipital Fusiform Cortex
iog_pats = [
    r"lateral occipital cortex.*inferior",
    r"occipital fusiform"
]

fus_idx = find_label_indices(labels, fus_pats)
itg_idx = find_label_indices(labels, itg_pats)
iog_idx = find_label_indices(labels, iog_pats)

if not fus_idx:
    raise ValueError("Could not find posterior/TO fusiform labels in HO labels.")
if not itg_idx:
    raise ValueError("Could not find posterior/TO inferior temporal labels in HO labels.")
if not iog_idx:
    raise ValueError("Could not find ventral occipital (IOG-approx) labels in HO labels.")

selected_label_indices = sorted(set(fus_idx + itg_idx + iog_idx))

vol_indices = []
for li in selected_label_indices:
    vi = ho_label_to_volume_index(labels, li)
    if vi is not None and vi >= 0:
        vol_indices.append(vi)

vol_indices = sorted(set(vol_indices))
max_vi = ho_data.shape[3] - 1
vol_indices = [vi for vi in vol_indices if 0 <= vi <= max_vi]
if not vol_indices:
    raise ValueError("Resolved HO volume indices are empty. Check label alignment.")

print("Selected HO labels (FG + ITG + ventral occipital):")
for li in selected_label_indices:
    print("  -", labels[li])
print("Resolved 4D volume indices:", vol_indices)
print("Probability threshold:", PROB_THR)

union_prob = np.max(ho_data[..., vol_indices], axis=3)
union_prob_img = nib.Nifti1Image(union_prob, ho_img.affine)


# ----------------------------
# 2) Resample probability to MNIPediatric target
# ----------------------------

union_prob_resamp = image.resample_to_img(
    union_prob_img,
    target_img,
    interpolation="continuous",
    force_resample=True,
    copy_header=True
)

prob_r = union_prob_resamp.get_fdata()


# ----------------------------
# 3) Threshold -> bilateral vOTC mask
# ----------------------------

votc_mask = (prob_r >= PROB_THR).astype(np.uint8)

# xyz boundaries - normally False
if APPLY_Z_CUTOFF or APPLY_Y_MIN or APPLY_Y_MAX:
    z_val = Z_MAX_MM if APPLY_Z_CUTOFF else None
    y_min_val = Y_MIN_MM if APPLY_Y_MIN else None
    y_max_val = Y_MAX_MM if APPLY_Y_MAX else None
    coord_mask = make_coord_mask_like(
        target_img,
        z_max=z_val,
        y_min=y_min_val,
        y_max=y_max_val
    )
    votc_mask = (votc_mask & coord_mask).astype(np.uint8)

votc_n_all = int(votc_mask.sum())
print("Bilateral vOTC voxels (before split):", votc_n_all)
if votc_n_all == 0:
    raise ValueError(
        "vOTC mask is empty. "
        "Try lowering PROB_THR or relaxing coordinate cutoffs."
    )

# seprate left and right hemi
left_hemi, right_hemi = make_hemisphere_masks(target_img)
votc_L = (votc_mask & left_hemi).astype(np.uint8)
votc_R = (votc_mask & right_hemi).astype(np.uint8)

votc_n_L = int(votc_L.sum())
votc_n_R = int(votc_R.sum())
print("Left vOTC voxels :", votc_n_L)
print("Right vOTC voxels:", votc_n_R)

thr_tag = str(PROB_THR).replace(".", "p")
base = f"vOTCcore_{NAME_TAG}_thr{thr_tag}_space-MNIPediatricAsym_cohort-3_res-2"

votc_L_path = os.path.join(OUTDIR, base + "_L_vOTC.nii.gz")
votc_R_path = os.path.join(OUTDIR, base + "_R_vOTC.nii.gz")

save_uint8_mask(votc_L, target_img, votc_L_path)
save_uint8_mask(votc_R, target_img, votc_R_path)

print("\nSaved vOTC masks:")
print("  L_vOTC:", votc_L_path)
print("  R_vOTC:", votc_R_path)

# Inspect Y-range of vOTC (for sanity)
ijk_all, xyz_all = get_world_xyz(target_img)
flat_v_L = votc_L.reshape(-1).astype(bool)
flat_v_R = votc_R.reshape(-1).astype(bool)
if flat_v_L.any():
    y_vals_L = xyz_all[flat_v_L, 1]
    print("Left vOTC Y-range :", float(y_vals_L.min()), "to", float(y_vals_L.max()))
if flat_v_R.any():
    y_vals_R = xyz_all[flat_v_R, 1]
    print("Right vOTC Y-range:", float(y_vals_R.min()), "to", float(y_vals_R.max()))


# ----------------------------
# 4) Define VWFA1/VWFA2 boxes (L & mirrored R) and intersect with vOTC
# ----------------------------

x = xyz_all[:, 0]
y = xyz_all[:, 1]
z = xyz_all[:, 2]

# left box
b1_L = VWFA1_BOUNDS_L
b2_L = VWFA2_BOUNDS_L

# right box
b1_R = mirror_x_bounds(b1_L)
b2_R = mirror_x_bounds(b2_L)

def box_mask(bounds, img_shape):
    bx = bounds["x"]
    by = bounds["y"]
    bz = bounds["z"]
    flat = (
        (x >= bx[0]) & (x <= bx[1]) &
        (y >= by[0]) & (y <= by[1]) &
        (z >= bz[0]) & (z <= bz[1])
    )
    return flat.reshape(img_shape).astype(np.uint8)

box1_L = box_mask(b1_L, target_img.shape[:3])
box2_L = box_mask(b2_L, target_img.shape[:3])
box1_R = box_mask(b1_R, target_img.shape[:3])
box2_R = box_mask(b2_R, target_img.shape[:3])

# in order our boxes no to cross the midline
box1_L = (box1_L & left_hemi).astype(np.uint8)
box2_L = (box2_L & left_hemi).astype(np.uint8)
box1_R = (box1_R & right_hemi).astype(np.uint8)
box2_R = (box2_R & right_hemi).astype(np.uint8)

# Intersection with vOTC
vwfa1_L = (votc_L & box1_L).astype(np.uint8)
vwfa2_L = (votc_L & box2_L).astype(np.uint8)
vwfa1_R = (votc_R & box1_R).astype(np.uint8)
vwfa2_R = (votc_R & box2_R).astype(np.uint8)

vwfa1_L_n = int(vwfa1_L.sum())
vwfa2_L_n = int(vwfa2_L.sum())
vwfa1_R_n = int(vwfa1_R.sum())
vwfa2_R_n = int(vwfa2_R.sum())

print("\nVWFA1 voxels (L, vOTC ∧ box1):", vwfa1_L_n)
print("VWFA2 voxels (L, vOTC ∧ box2):", vwfa2_L_n)
print("VWFA1 voxels (R, vOTC ∧ box1):", vwfa1_R_n)
print("VWFA2 voxels (R, vOTC ∧ box2):", vwfa2_R_n)

if vwfa1_L_n == 0:
    print("WARNING: Left VWFA1 mask is empty. Check bounds or vOTC extent.")
if vwfa2_L_n == 0:
    print("WARNING: Left VWFA2 mask is empty. Check bounds or vOTC extent.")
if vwfa1_R_n == 0:
    print("WARNING: Right VWFA1 mask is empty. Check bounds or vOTC extent.")
if vwfa2_R_n == 0:
    print("WARNING: Right VWFA2 mask is empty. Check bounds or vOTC extent.")

vwfa1_L_path = os.path.join(OUTDIR, base + "_L_VWFA1.nii.gz")
vwfa2_L_path = os.path.join(OUTDIR, base + "_L_VWFA2.nii.gz")
vwfa1_R_path = os.path.join(OUTDIR, base + "_R_VWFA1.nii.gz")
vwfa2_R_path = os.path.join(OUTDIR, base + "_R_VWFA2.nii.gz")

save_uint8_mask(vwfa1_L, target_img, vwfa1_L_path)
save_uint8_mask(vwfa2_L, target_img, vwfa2_L_path)
save_uint8_mask(vwfa1_R, target_img, vwfa1_R_path)
save_uint8_mask(vwfa2_R, target_img, vwfa2_R_path)

print("\nSaved VWFA masks:")
print("  L_VWFA1:", vwfa1_L_path)
print("  L_VWFA2:", vwfa2_L_path)
print("  R_VWFA1:", vwfa1_R_path)
print("  R_VWFA2:", vwfa2_R_path)

print("\nAll ROIs are in:", OUTDIR)
print("Jobs done, go take a coffee, a big one!")
