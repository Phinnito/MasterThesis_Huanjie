#!/bin/bash

# =========================================================================
# SCRIPT 1: RUN FMRIPREP BATCH
# Purpose: Cleans temporary directories, runs fMRIPrep for all subjects,
#          and provides a file count summary upon completion.
# Usage: ./run_fmriprep_batch.sh <sub-id-1> <sub-id-2> ...
# =========================================================================

# ========================== CONFIGURATION ================================
# --- Final Data Storage Location (Network Server) ---
BIDS_DIR="/your/path/"

# --- Local Temporary Processing Location ---
TMP_DIR="/mnt2/fmri_prep_tmp"
TMP_OUTPUT_DIR="${TMP_DIR}/output"
TMP_WORK_DIR="${TMP_DIR}/work"

# --- Freesurfer License ---
FS_LICENSE="/home/ubuntu/license.txt"/your/path/fmriprep/run_fmriprep_1.sh

# ======================== VALIDATE INPUT ============================
if [ -z "$@" ]; then
    echo "Usage: $0 <subject_label_1> <subject_label_2> ..."
    echo "Error: At least one subject label must be provided."
    exit 1
fi

SUBJECT_LABELS="$@"
echo "========================================="
echo "The following subjects will be processed: ${SUBJECT_LABELS}"
echo "========================================="

# ========================= SCRIPT EXECUTION ==============================
# --- 1. Clean and Prepare Main Temporary Directories at the Start ---
echo "Cleaning up temporary directories before starting the batch..."
rm -rf "${TMP_OUTPUT_DIR}"
rm -rf "${TMP_WORK_DIR}"
mkdir -p "${TMP_OUTPUT_DIR}"
mkdir -p "${TMP_WORK_DIR}"
chmod -R a+rwX "$TMP_DIR"
echo "Temporary directories are ready."

# --- 2. Loop Through Subjects and Run fMRIPrep ---
for SUBJECT_LABEL in $SUBJECT_LABELS; do

    echo ""
    echo "#################################################################"
    echo "### STARTING FMRIPREP FOR sub-${SUBJECT_LABEL}"
    echo "#################################################################"
    date

    SUB_WORK_DIR="${TMP_WORK_DIR}/sub-${SUBJECT_LABEL}"
    mkdir -p "$SUB_WORK_DIR"

    echo "Local work directory: ${SUB_WORK_DIR}"
    echo "Local output directory: ${TMP_OUTPUT_DIR} (accumulating results)"

    # Run FMRIPREP
    fmriprep-docker \
      --user=$(id -u):$(id -g) \
      "$BIDS_DIR" \
      "$TMP_OUTPUT_DIR" \
      --participant-label "$SUBJECT_LABEL" \
      --work-dir "$SUB_WORK_DIR" \
      --fs-license-file "$FS_LICENSE" \
      --bold2anat-init t1w \
      --output-layout legacy \
      --output-spaces MNI152NLin2009cAsym MNIPediatricAsym:cohort-3:res-2 T1w fsaverage5 fsnative \
      --fd-spike-threshold 1.5

    fmriprep_exit_code=$?
    if [ $fmriprep_exit_code -ne 0 ]; then
        echo "ERROR: fMRIPrep failed for sub-${SUBJECT_LABEL} with exit code ${fmriprep_exit_code}."
        echo "Please check the log files. Aborting the entire batch."
        exit 1
    fi

    echo "fMRIPrep finished successfully for sub-${SUBJECT_LABEL}."
    
    rm -rf "$SUB_WORK_DIR"
done


# --- 3. Generate File Count Summary ---
echo ""
echo "================================================================="
echo "### GENERATING FILE COUNT SUMMARY ###"
echo "================================================================="

for SUBJECT_LABEL in $SUBJECT_LABELS; do
    echo "--- Summary for sub-${SUBJECT_LABEL} ---"
    
    # --- Corrected Paths to check ---
    # Path for main fmriprep derivatives (anat, func)
    SUB_DERIV_DIR="${TMP_OUTPUT_DIR}/fmriprep/sub-${SUBJECT_LABEL}"
    
    # Path for freesurfer derivatives
    # Freesurfer output is at the same level as the fmriprep directory, not inside it.
    SUB_FS_DIR="${TMP_OUTPUT_DIR}/freesurfer/sub-${SUBJECT_LABEL}"

    # --- Check main derivatives (anat, func) ---
    if [ -d "$SUB_DERIV_DIR" ]; then
        for FOLDER in anat func; do
            FOLDER_PATH="${SUB_DERIV_DIR}/${FOLDER}"
            if [ -d "${FOLDER_PATH}" ]; then
                # Use find to count files, not directories
                count=$(find "${FOLDER_PATH}" -type f | wc -l)
                echo "  fmriprep/${FOLDER}: ${count} files"
            else
                echo "  fmriprep/${FOLDER}: NOT FOUND"
            fi
        done
    else
        echo "  Directory fmriprep/sub-${SUBJECT_LABEL}: NOT FOUND"
    fi

    # --- Check freesurfer derivatives ---
    if [ -d "$SUB_FS_DIR" ]; then
        for FOLDER in surf mri label; do
            FOLDER_PATH="${SUB_FS_DIR}/${FOLDER}"
            if [ -d "${FOLDER_PATH}" ]; then
                # Use find to count files, not directories
                count=$(find "${FOLDER_PATH}" -type f | wc -l)
                echo "  freesurfer/${FOLDER}: ${count} files"
            else
                echo "  freesurfer/${FOLDER}: NOT FOUND"
            fi
        done
    else
        echo "  Directory freesurfer/sub-${SUBJECT_LABEL}: NOT FOUND"
    fi
done


echo ""
echo "================================================================="
echo "### BATCH PROCESSING COMPLETE! ###"
echo "All subjects have been processed. Please review the summary above."
echo ""
echo "Now, you can run the second script to package the results:"
echo "./run_fmriprep_2.sh ${SUBJECT_LABELS}"
echo "================================================================="
