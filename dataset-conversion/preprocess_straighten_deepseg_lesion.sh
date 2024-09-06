#!/bin/bash
#
# Preprocess data.
#
# Dependencies (versions):
# - SCT (5.4.0)
#
# Usage:
# sct_run_batch -script preprocess_data.sh -path-data <PATH-TO-DATASET> -path-output <PATH-TO-OUTPUT> -jobs <num-cpu-cores>

# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/ses-0X/anat/

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Global variables
CENTERLINE_METHOD="svm"  # method sct_deepseg_sc uses for centerline extraction: 'svm', 'cnn'

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT


# CONVENIENCE FUNCTIONS
# ======================================================================================================================

# Retrieve input params and other params
PATH_DATA_DIR=$1
PATH_DEST_DIR="/home/GRAMES.POLYMTL.CA/u114716/tum-poly/prepro_straighten_deepseg_lesion"
PATH_DATA_PROCESSED="${PATH_DEST_DIR}/data_processed"
PATH_DATA_PROCESSED_CLEAN="${PATH_DEST_DIR}/data_processed_clean"

if [[ ! -d ${PATH_DATA_PROCESSED} ]]; then
  mkdir -p ${PATH_DATA_PROCESSED}
fi

if [[ ! -d ${PATH_DATA_PROCESSED_CLEAN} ]]; then
  mkdir -p ${PATH_DATA_PROCESSED_CLEAN}/imagesTs_straight
  mkdir -p ${PATH_DATA_PROCESSED_CLEAN}/labelsTs_straight
  mkdir -p ${PATH_DATA_PROCESSED_CLEAN}/warpingTs_straight_to_native
fi

# get starting time:
start=`date +%s`

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# # Go to folder where data will be copied and processed
# cd $PATH_DATA_PROCESSED

# Copy source images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -avzh $PATH_DATA_DIR/imagesTs_deepseg/* $PATH_DATA_PROCESSED/imagesTs_deepseg/

# Copy segmentation ground truths (GT)
# mkdir -p derivatives/labels
rsync -avzh $PATH_DATA_DIR/labelsTs_deepseg/* $PATH_DATA_PROCESSED/labelsTs_deepseg/

# print the current contents of the PATH_DATA_PROCESSED folder
echo "The contents of the PATH_DATA_PROCESSED folder are:"
ls -l

cd $PATH_DATA_PROCESSED

for label_file in ${PATH_DATA_PROCESSED}/labelsTs_deepseg/*.nii.gz; do
  # Go to subject folder for source images
  file=$(basename $label_file .nii.gz)

  # Define variable for SC segmentation
  file_seg=${file}_seg-manual

  # Get the SC seg from the region-based label
  sct_maths -i labelsTs_deepseg/${file}.nii.gz -bin 0 -o labelsTs_deepseg/${file_seg}.nii.gz

  # Define variable for lesion mask
  file_lesion=${file}_lesion-manual

  # Get the lesion seg from the region-based label
  sct_maths -i labelsTs_deepseg/${file}.nii.gz -bin 1 -o labelsTs_deepseg/${file_lesion}.nii.gz

  # # NOTE: the sform and qform matrices don't match for a few subjects (for both images and SC seg), setting sform to qform
  # sct_image -i ${file}.nii.gz -set-sform-to-qform -o ${file}.nii.gz
  # sct_image -i ${file_seg}.nii.gz -set-sform-to-qform -o ${file_seg}.nii.gz

  # Straighten the SC
  sct_straighten_spinalcord -i imagesTs_deepseg/${file}_0000.nii.gz -s labelsTs_deepseg/${file_seg}.nii.gz -o imagesTs_deepseg/${file}_straight_0000.nii.gz

  # Straighten the SC mask using straightened SC as reference
  sct_apply_transfo -i labelsTs_deepseg/${file_seg}.nii.gz -d imagesTs_deepseg/${file}_straight_0000.nii.gz -w warp_curve2straight.nii.gz -x linear -o labelsTs_deepseg/${file_seg}_straight.nii.gz
  # Straighten the lesion mask using straightened SC as reference
  sct_apply_transfo -i labelsTs_deepseg/${file_lesion}.nii.gz -d imagesTs_deepseg/${file}_straight_0000.nii.gz -w warp_curve2straight.nii.gz -x linear -o labelsTs_deepseg/${file_lesion}_straight.nii.gz

  # Binarize the lesion and SC masks
  sct_maths -i labelsTs_deepseg/${file_seg}_straight.nii.gz -bin 0.5 -o labelsTs_deepseg/${file_seg}_straight.nii.gz
  sct_maths -i labelsTs_deepseg/${file_lesion}_straight.nii.gz -bin 0.5 -o labelsTs_deepseg/${file_lesion}_straight.nii.gz

  file_label=${file}_straight
  # Add lesion seg to the SC seg to create region-based labels
  sct_maths -i labelsTs_deepseg/${file_seg}_straight.nii.gz -add labelsTs_deepseg/${file_lesion}_straight.nii.gz -o labelsTs_deepseg/${file_label}.nii.gz

  # Copy the straightened images to the clean folder
  rsync -avzh imagesTs_deepseg/${file}_straight_0000.nii.gz ${PATH_DATA_PROCESSED_CLEAN}/imagesTs_straight/${file}_straight_0000.nii.gz

  # Copy the straightened labels to the clean folder
  rsync -avzh labelsTs_deepseg/${file_label}.nii.gz ${PATH_DATA_PROCESSED_CLEAN}/labelsTs_straight/${file_label}.nii.gz

  # Copy the warping files to the clean folder
  # rsync -avzh warp_curve2straight.nii.gz ${PATH_DATA_PROCESSED_CLEAN}/warpingTs_straight_to_native/${file}_warp_curve2straight.nii.gz
  rsync -avzh warp_straight2curve.nii.gz ${PATH_DATA_PROCESSED_CLEAN}/warpingTs_straight_to_native/${file}_warp_straight2curve.nii.gz

done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"