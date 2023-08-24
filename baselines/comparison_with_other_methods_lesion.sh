#!/bin/bash
#
# Compare our nnUNet model with other methods (nnUNet-2D and nnUNet-3D) on sci-zurich and
# sci-colorado datasets
# Note: subjects from both datasets have to be located in the same BIDS-like folder, example:
# ├── derivatives
# │	 └── labels
# │	     ├── sub-5416   # sci-colorado subject
# │	     │	 └── anat
# │	     │	     ├── sub-5416_T2w_lesion-manual.json
# │	     │	     ├── sub-5416_T2w_lesion-manual.nii.gz
# │	     │	     ├── sub-5416_T2w_seg-manual.json
# │	     │	     └── sub-5416_T2w_seg-manual.nii.gz
# │	     └── sub-zh01   # sci-zurich subject
# │	         └── ses-01
# │	             └── anat
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.json
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_lesion-manual.nii.gz
# │	                 ├── sub-zh01_ses-01_acq-sag_T2w_seg-manual.json
# │	                 └── sub-zh01_ses-01_acq-sag_T2w_seg-manual.nii.gz
# ├── sub-5416    # sci-colorado subject
# │	 └── anat
# │	     ├── sub-5416_T2w.json
# │	     └── sub-5416_T2w.nii.gz
# └── sub-zh01    # sci-zurich subject
#    └── ses-01
#        └── anat
#            ├── sub-zh01_ses-01_acq-sag_T2w.json
#            └── sub-zh01_ses-01_acq-sag_T2w.nii.gz
#
# Note: conda environment with nnUNetV2 is required to run this script.
# For details how to install nnUNetV2, see:
# https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md#installation
#
# Usage:
#     sct_run_batch -config config.json
#
# Example of config.json:
# {
#  "path_data"   : "<PATH_TO_DATASET>",
#  "path_output" : "<PATH_TO_DATASET>_2023-08-18",
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/baselines/comparison_with_other_methods_lesion.sh",
#  "jobs"        : 8,
#  "script_args" : "<PATH_TO_REPO>/model_seg_sci/packaging/run_inference_single_subject.py <PATH_TO_MODEL>/sci-multisite-model"
# }
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Author: Jan Valosek (adapted by Naga Karthik for lesions)
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from the sct_run_batch script to the log (to allow easier debug)
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

SUBJECT=$1
PATH_NNUNET_SCRIPT=$2
PATH_NNUNET_MODEL=$3

echo "SUBJECT: ${SUBJECT}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------
# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

# Segment lesion using our nnUNet model
segment_lesion_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d

  FILELESION="${file}_lesion_nnunet_${kernel}"

  # get the GT sc seg to be used for QC
  FILEGTSEG="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_seg-manual"

  # Get the start time
  start_time=$(date +%s)
  # Run lesion segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILELESION}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNet_${kernel} -pred-type lesion
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILELESION},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  if [[ $SUBJECT =~ "sub-zh" ]]; then
    # Generate sagittal QC report
    sct_qc -i ${file}.nii.gz -s ${FILEGTSEG}.nii.gz -d ${FILELESION}.nii.gz -p sct_deepseg_lesion -plane sagittal -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    # Generate axial QC report
    sct_qc -i ${file}.nii.gz -s ${FILEGTSEG}.nii.gz -d ${FILELESION}.nii.gz -p sct_deepseg_lesion -plane axial -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
  # Compute ANIMA segmentation performance metrics
  compute_anima_metrics ${FILELESION} ${file}_lesion-manual.nii.gz
}

# Compute ANIMA segmentation performance metrics
compute_anima_metrics(){
  # We have to copy qform matrix from seg-manual to the automatically generated segmentation to avoid ITK error:
  # "Description: ITK ERROR: SegmentationMeasuresImageFilter(): Inputs do not occupy the same physical space!"
  # Related to the following issue : https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135
  sct_image -i ${file}_lesion-manual.nii.gz -copy-header ${FILELESION}.nii.gz -o ${FILELESION}_updated_header.nii.gz

  # Compute ANIMA segmentation performance metrics
  # -i : input segmentation
  # -r : GT segmentation
  # -o : output file
  # -d : surface distances evaluation
  # -s : compute metrics to evaluate a segmentation
  # -l : lesion detection evaluation
  # -X : stores results into a xml file.
  ${anima_binaries_path}/animaSegPerfAnalyzer -i ${FILELESION}_updated_header.nii.gz -r ${file}_lesion-manual.nii.gz -o ${PATH_RESULTS}/${FILELESION} -d -s -l -X

  rm ${FILELESION}_updated_header.nii.gz
}

# Copy GT segmentation
copy_gt(){
  local file="$1"
  # Construct file name to GT segmentation located under derivatives/labels
  FILELESIONMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_lesion-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILELESIONMANUAL"
  if [[ -e $FILELESIONMANUAL ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILELESIONMANUAL ${file}_lesion-manual.nii.gz
  else
      echo "File ${FILELESIONMANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual GT segmentation ${FILELESIONMANUAL}.nii.gz does not exist. Exiting."
      exit 1
  fi
}

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# get starting time:
start=`date +%s`

# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source T2w images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
if [[ $SUBJECT =~ "sub-zh" ]]; then
  # for sci-zurich, copy only sagittal T2w to save space
  rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*sag_T2w.* .
else
  rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*T2w.* .
fi

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# sci-zurich
if [[ $SUBJECT =~ "sub-zh" ]]; then
    # We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
    file_t2="${SUBJECT//[\/]/_}"_acq-sag_T2w
# sci-colorado
else
    file_t2="${SUBJECT}"_T2w
fi

# Copy GT segmentation
copy_gt "${file_t2}"

# binarize the GT with thr 0.5
sct_maths -i ${file_t2}_lesion-manual.nii.gz -bin 0.5 -o ${file_t2}_lesion-manual.nii.gz

# Check if file_t2 exists
if [[ ! -e ${file_t2}.nii.gz ]]; then
    echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
    echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
    exit 1
else
    # Segment lesion using different methods and compute ANIMA segmentation performance metrics
    segment_lesion_nnUNet "${file_t2}" '2d'
    segment_lesion_nnUNet "${file_t2}" '3d'
fi

# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------

# Display results (to easily compare integrity across SCT versions)
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
