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
DATASET=$2
PATH_STACKING_SCRIPT=$3
PATH_NNUNET_SCRIPT="/home/GRAMES.POLYMTL.CA/u114716/tum-poly/miccai_amai_lesions_spine/packaging/run_inference_single_subject.py"

echo "SUBJECT: ${SUBJECT}"
echo "DATASET: ${DATASET}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
# echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------
# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

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


# Segment spinal cord using methods available in SCT (sct_deepseg_sc or sct_propseg)
segment_lesion() {
  local file="$1"
  local contrast="$2"
  local model_type="$3"   # single_site or two_site
  local kernel="2d"   # using 2D for TUM models by default

  # Segment lesion
  # FILELESION="${file}_lesion_deepseg"
  FILELESION="${file}_lesion_tum_${model_type}"

  if [[ $model_type == 'single_site' ]]; then
    PATH_NNUNET_MODEL="/home/GRAMES.POLYMTL.CA/u114716/nnunet-v2/nnUNet_results/Dataset901_tumMSChunksRegion"
  elif [[ $model_type == 'two_site' ]]; then
    PATH_NNUNET_MODEL="/home/GRAMES.POLYMTL.CA/u114716/nnunet-v2/nnUNet_results/Dataset910_tumMSChunksPolyNYUAxialRegion"
  fi

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  # sct_deepseg_lesion -i ${file}.nii.gz -ofolder . -c ${contrast} -centerline svm
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILELESION}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel} -pred-type lesion -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILELESION},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

}


# Segment lesion using our nnUNet model
segment_ms_lesion_agnostic(){
  local file="$1"
  # local kernel="$2"     # 2d or 3d

  FILELESION="${file}_lesion_seg_plb"
  # FILELESION="${file}_lesion_seg_tum"

  # Get the start time
  start_time=$(date +%s)
  # Run lesion segmentation
  # python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILELESION}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel} -pred-type lesion
  SCT_USE_GPU=1 sct_deepseg -task seg_ms_lesion -i ${file}.nii.gz -o ${FILELESION}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # SCT_USE_GPU=1 sct_deepseg -task seg_sc_ms_lesion_axial_t2w -i ${file}.nii.gz -o ${FILELESION}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILELESION},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # # Compute ANIMA segmentation performance metrics
  # compute_anima_metrics ${FILELESION} ${file}_lesion-manual.nii.gz
}

# Copy GT lesion segmentation (located under derivatives/labels)
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
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_acq-ax_*T2w.nii.gz .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------

if [[ $DATASET == "bavaria-quebec-spine-ms-unstitched" ]]; then

  chunks="chunk-1 chunk-2 chunk-3"
  # Loop across chunks
  for chunk in ${chunks}; do
    # We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
    file_t2=${SUBJECT//[\/]/_}_acq-ax_${chunk}_T2w

    # Copy GT lesion segmentation
    copy_gt "${file_t2}"

    # # binarize the GT with thr 0.5
    # sct_maths -i ${file_t2}_lesion-manual.nii.gz -bin 0.5 -o ${file_t2}_lesion-manual.nii.gz

    # Check if file_t2 exists
    if [[ ! -e ${file_t2}.nii.gz ]]; then
        echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
        echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
        exit 1
    fi

    # Segment lesion using different methods and compute ANIMA segmentation performance metrics
    # ms-lesion-agnostic modle
    CUDA_VISIBLE_DEVICES=3 segment_ms_lesion_agnostic "${file_t2}"
    # # TUM axial T2w MS lesion seg model
    # CUDA_VISIBLE_DEVICES=3 segment_lesion "${file_t2}" 't2_ax' 'single_site'
    # CUDA_VISIBLE_DEVICES=3 segment_lesion "${file_t2}" 't2_ax' 'two_site'
    # # sct_deepseg_lesion model
    # segment_lesion ${file_t2} 't2_ax'
  done

  # find . -type f -name "*_seg.nii.gz" -exec rm {} \;

  echo "====================================================================================================="
  echo "Stacking individual chunks of prediction and reference ..."
  echo "====================================================================================================="

  # Stack individual chunk predictions
  python ${PATH_STACKING_SCRIPT} -prediction ${PATH_DATA_PROCESSED}/${SUBJECT}/anat -reference ${PATH_DATA_PROCESSED}/${SUBJECT}/anat

  # # Compute ANIMA segmentation performance metrics
  # compute_anima_metrics ${FILELESION} ${file}_lesion-manual.nii.gz


elif [[ $DATASET == "sct-testing-large" ]]; then 

  # We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
  file_t2=${SUBJECT//[\/]/_}_acq-ax_T2w

  # Copy GT lesion segmentation
  copy_gt "${file_t2}"

  # binarize the GT with thr 0.5
  sct_maths -i ${file_t2}_lesion-manual.nii.gz -bin 0.5 -o ${file_t2}_lesion-manual.nii.gz

  # Check if file_t2 exists
  if [[ ! -e ${file_t2}.nii.gz ]]; then
      echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
      exit 1
  fi

  # Segment lesion using different methods and compute ANIMA segmentation performance metrics
  CUDA_VISIBLE_DEVICES=2 segment_ms_lesion_agnostic "${file_t2}"
  # # TUM axial T2w MS lesion seg model
  # CUDA_VISIBLE_DEVICES=3 segment_lesion "${file_t2}" 't2_ax' 'single_site'
  # CUDA_VISIBLE_DEVICES=3 segment_lesion "${file_t2}" 't2_ax' 'two_site'
  # segment_lesion ${file_t2} 't2_ax'

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
