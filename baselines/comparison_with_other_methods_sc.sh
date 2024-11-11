#!/bin/bash
#
# Compare our nnUNet model with other methods (sct_propseg, sct_deepseg_sc 2d, sct_deepseg_sc 3d,
# MONAI contrast-agnostic (part of SCT v6.2)) on bavaria-quebec-spine-ms-unstitched dataset.
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
#  "script"      : "<PATH_TO_REPO>/model_seg_sci/baselines/comparison_with_other_methods.sh",
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
# Author: Jan Valosek
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
PATH_NNUNET_SCRIPT=$2   # path to the nnUNet SCIseg run_inference_single_subject.py
PATH_NNUNET_MODEL=$3    # path to the nnUNet SCIseg model

echo "SUBJECT: ${SUBJECT}"
echo "PATH_NNUNET_SCRIPT: ${PATH_NNUNET_SCRIPT}"
echo "PATH_NNUNET_MODEL: ${PATH_NNUNET_MODEL}"

# ------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------------------
# Get ANIMA binaries path
anima_binaries_path=$(grep "^anima = " ~/.anima/config.txt | sed "s/.* = //" | sed 's/\/$//')

# Segment spinal cord using methods available in SCT (sct_deepseg_sc or sct_propseg)
segment_sc() {
  local file="$1"
  local contrast="$2"
  local method="$3"     # deepseg or propseg
  local kernel="$4"     # 2d or 3d; only relevant for deepseg

  # Segment spinal cord
  if [[ $method == 'deepseg' ]];then
      FILESEG="${file}_seg_${method}_${kernel}"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      sct_deepseg_sc -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -kernel ${kernel} -qc ${PATH_QC} -qc-subject ${SUBJECT}
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  elif [[ $method == 'propseg' ]]; then
      FILESEG="${file}_seg_${method}"

      # Get the start time
      start_time=$(date +%s)
      # Run SC segmentation
      sct_propseg -i ${file}.nii.gz -o ${FILESEG}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
      # Get the end time
      end_time=$(date +%s)
      # Calculate the time difference
      execution_time=$(python3 -c "print($end_time - $start_time)")
      echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

      # Remove centerline (we don't need it)
      rm ${file}_centerline.nii.gz
  fi
}

# Segment spinal cord using our SCIseg nnUNet model
segment_sc_nnUNet(){
  local file="$1"
  local kernel="$2"     # 2d or 3d

  FILESEG="${file}_seg_nnunet_${kernel}"

  # Get the start time
  start_time=$(date +%s)
  # Run SC segmentation
  python ${PATH_NNUNET_SCRIPT} -i ${file}.nii.gz -o ${FILESEG}.nii.gz -path-model ${PATH_NNUNET_MODEL}/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__${kernel} -pred-type sc -use-gpu
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
}

# Segment spinal cord using the MONAI contrast-agnostic model (part of SCT v6.2)
segment_sc_MONAI(){
  local file="$1"
  local version="$2"     # v2.0 or v2.5

  FILESEG="${file}_seg_monai_${version}"

  # Get the start time
  start_time=$(date +%s)
  if [[ $version == 'v2.6' ]]; then
    # Run SC segmentation
    PATH_MODEL="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/saved_models/lifelong/nnunet-plain_seed=50_ndata=15_ncont=11_nf=384_opt=adam_lr=0.001_AdapW_bs=2_20241018-0508"
    PATH_MONAI_SCRIPT="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/contrast-agnostic-softseg-spinalcord/monai/run_inference_single_image.py"
    python ${PATH_MONAI_SCRIPT} --path-img ${file}.nii.gz --path-out . --chkp-path ${PATH_MODEL} --device gpu --model monai --pred-type soft --pad-mode edge --max-feat 384
    # Rename MONAI output
    mv ${file}_pred.nii.gz ${FILESEG}.nii.gz
    # Binarize MONAI output (which is soft by default); output is overwritten
    sct_maths -i ${FILESEG}.nii.gz -bin 0.5 -o ${FILESEG}.nii.gz

  else
    # Run SC segmentation using SCT v6.2
    sct_deepseg -task seg_sc_contrast_agnostic -i ${file}.nii.gz -o ${FILESEG}.nii.gz
  fi
  # Get the end time
  end_time=$(date +%s)
  # Calculate the time difference
  execution_time=$(python3 -c "print($end_time - $start_time)")
  echo "${FILESEG},${execution_time}" >> ${PATH_RESULTS}/execution_time.csv

  # Generate QC report
  sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
}

# Copy GT spinal cord segmentation (located under derivatives/labels)
copy_gt(){
  local file="$1"
  # Construct file name to GT segmentation located under derivatives/labels
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_seg-manual.nii.gz"
  echo ""
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
      echo "Found! Copying ..."
      rsync -avzh $FILESEGMANUAL ${file}_seg-manual.nii.gz
  else
      echo "File ${FILESEGMANUAL}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
      echo "ERROR: Manual GT segmentation ${FILESEGMANUAL}.nii.gz does not exist. Exiting."
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

# custom_url="https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/download/v2.0/model_2023-09-18.zip"
# sct_deepseg -install seg_sc_contrast_agnostic -custom-url ${custom_url}

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source T2w images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*ax_chunk-*.nii.gz .
# rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT//[\/]/_}_*ax_T2w.nii.gz .

# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# run a counter from 0 to 4 for the chunks
for i in {1..3}; do
# for i in {1}; do
    # Define T2w file
    file_t2="${SUBJECT//[\/]/_}_acq-ax_chunk-${i}_T2w"
    # file_t2="${SUBJECT//[\/]/_}_acq-ax_T2w"

    # Copy GT spinal cord segmentation
    copy_gt "${file_t2}"

    # Check if file_t2 exists
    if [[ ! -e ${file_t2}.nii.gz ]]; then
        echo "File ${file_t2}.nii.gz does not exist" >> ${PATH_LOG}/missing_files.log
        echo "ERROR: File ${file_t2}.nii.gz does not exist. Exiting."
        exit 1
    fi

    # Segment SC using different methods and compute ANIMA segmentation performance metrics
    CUDA_VISIBLE_DEVICES=2 segment_sc_MONAI "${file_t2}" 'v2.6'
    CUDA_VISIBLE_DEVICES=3 segment_sc_MONAI "${file_t2}" 'v2.5'
    # CUDA_VISIBLE_DEVICES=3 segment_sc_nnUNet "${file_t2}" '3d_fullres'
    # # segment_sc_nnUNet "${file_t2}" '3d_fullres'
    segment_sc "${file_t2}" 't2' 'propseg'
    segment_sc "${file_t2}" 't2' 'deepseg' '2d'
    # segment_sc "${file_t2}" 't2' 'deepseg' '3d'

    
done

# methods_to_compare="propseg deepseg_2d deepseg_3d monai_v2.0 nnunet_2d"
methods_to_compare="propseg deepseg_2d monai_v2.5 monai_v2.6" # monai_v2.4"

# mkdir -p ${PATH_RESULTS}/GTs
# # copy the GTs to the results folder
# rsync -avzh ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${SUBJECT//[\/]/_}_*ax_T2w_seg-manual.nii.gz ${PATH_RESULTS}/GTs

# for method in ${methods_to_compare}; do
#     # create multiple sub-folders within the results folder
#     mkdir -p ${PATH_RESULTS}/preds/${method}
#     # Define T2w file
#     file_t2="${SUBJECT//[\/]/_}_acq-ax_T2w"
#     # copy the predictions to the results folder
#     rsync -avzh ${file_t2}_seg_${method}.nii.gz ${PATH_RESULTS}/preds/${method}
# done

for i in {1..3}; do

  mkdir -p ${PATH_RESULTS}/GTs/chunk-${i}
  # copy the GTs to the results folder
  rsync -avzh ${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${SUBJECT//[\/]/_}_*ax_chunk-${i}_T2w_seg-manual.nii.gz ${PATH_RESULTS}/GTs/chunk-${i}

  for method in ${methods_to_compare}; do
      # create multiple sub-folders within the results folder
      mkdir -p ${PATH_RESULTS}/preds/${method}/chunk-${i}
      # Define T2w file
      file_t2="${SUBJECT//[\/]/_}_acq-ax_chunk-${i}_T2w"
      # copy the predictions to the results folder
      rsync -avzh ${file_t2}_seg_${method}.nii.gz ${PATH_RESULTS}/preds/${method}/chunk-${i}
  done

done

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
